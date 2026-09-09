"""MetaPlannerLoop — the agent-driven scratchpad loop (outer ``g → f``).

This replaces the former verdict-driven cross-round orchestrator.  Per
``continuation-verdict-semantics.md`` §5.2 and ``meta-plan-prompt-and-ppf.md``
§3, the loop is:

::

    while agent has not decided to stop:            # stop is FREE
        plan = scratchpad.submit_plan(agent)        # hard gate — MUST submit
        pef = g(plan + PPF)                         # build_pipeline
        artifact = session.run(pef)                 # f: execute
        goal_status = session.evaluate(pef)         # framework fact layer
        scratchpad.record_round(actual=…)           # framework writes (read-only)
        # current round injected expanded for the agent to audit;
        # agent rewrites plan (overwrite) or keeps it (resubmit verbatim)

Stopping is free (the agent decides); continuing requires a plan.  The
framework keeps one safety net orthogonal to free stop: ``max_rounds`` /
budget exceeded → forced stop with the half-finished report + ``goal_status``.
The net prevents "the agent never decides"; it does not make the decision for
the agent.

No verdict (SAT/UNSAT/CONTINUE) is read to drive the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from quro.core.catalog_adapter import StepCatalogAdapter
from quro.core.conformance import ConformanceChecker, SessionDependencies
from quro.core.cross_round_handoff import CrossRoundHandoff
from quro.core.domain.step_type import StepTypeCatalog
from quro.core.round_controller import PlanGate as ControllerPlanGate
from quro.core.round_controller import RoundController
from quro.core.step_builder import StepBuilder
from quro.core.testing import InMemoryAuditLog
from quro.planner.domain import PlannerDomain
from quro.planner.scratchpad import PPF, PlanGate, RoundRecord, Scratchpad
from quro.planner.session import Session
from quro.planner.session_journal import IMetaPlannerSessionJournal

logger = logging.getLogger(__name__)


@dataclass
class MetaPlannerLoopConfig:
    """Configuration for the cross-round scratchpad loop.

    Args:
        max_rounds: Safety bound on the number of rounds.  This is a *guard*,
            not a convergence semantic — it prevents "the agent never decides",
            but never makes the decision for the agent.
        debug: Enable debug logging.
    """

    max_rounds: int = 8
    debug: bool = False


@dataclass
class MetaPlannerLoopResult:
    """The loop's output: the report + fact layer, no verdict."""

    rounds: int = 0
    report: str = ""
    goal_status: dict[str, Any] = field(default_factory=dict)
    conclusion: str | None = None
    aborted: bool = False
    """True when the safety net (``max_rounds``) forced a stop."""


class MetaPlannerLoop:
    """Drive the agent-driven scratchpad loop across rounds.

    Args:
        planner: The ``PlannerDomain`` (HTN ``g : Problem -> [StepSpec]``).
        step_types: Step types available for execution.
        backend: LLM backend for MetaPlanner calls (the agent).
        step_executor: StepExecutor (e.g. ``StepAdapter``) for execution.
        script: The framework-side single-round primitive.
        scratchpad: The agent's planning surface.
        pipeline_config: PipelineConfig for the MetaPlanner tool set.
        config: Optional ``MetaPlannerLoopConfig``.
    """

    def __init__(
        self,
        *,
        planner: PlannerDomain,
        step_types: StepTypeCatalog,
        backend: Any,
        step_executor: Any,
        session: Session,
        scratchpad: Scratchpad,
        pipeline_config: Any,
        config: MetaPlannerLoopConfig | None = None,
    ) -> None:
        self._planner = planner
        self._step_types = step_types
        self._backend = backend
        self._step_executor = step_executor
        self._session = session
        self._scratchpad = scratchpad
        self._pipeline_config = pipeline_config
        self._config = config or MetaPlannerLoopConfig()
        self._handoff = CrossRoundHandoff()
        self._audit_log = InMemoryAuditLog()

        # Conformance check: validate wiring at construction time.
        checker = ConformanceChecker(testing=True)
        violations = checker.check_wiring(SessionDependencies(
            builder=StepBuilder(step_types=self._step_types),
            store=_NoopStore(),
            audit_log=self._audit_log,
            controller=self._session,
            handoff=self._handoff,
        ))
        if violations:
            violation_msgs = "; ".join(v.message for v in violations)
            logger.warning("Conformance wiring violations: %s", violation_msgs)

    # ------------------------------------------------------------------- run
    def run(
        self,
        problem: str,
        goal_facts: list[str],
    ) -> MetaPlannerLoopResult:
        """Run the agent-driven loop until the agent stops (or the guard hits).

        Args:
            problem: The original problem statement (root context).
            goal_facts: The original SUCCESS CRITERIA (fact names).

        Returns:
            ``MetaPlannerLoopResult`` — the report + goal_status + conclusion.
        """
        ppf = PPF(problem=problem, goals=list(goal_facts))
        self._scratchpad.open(ppf)

        for round_idx in range(self._config.max_rounds):
            # 1. Plan gate (required): the agent MUST submit a plan to continue.
            plan_text, stop_requested = self._plan_round(round_idx)
            if stop_requested:
                return self._finish(rounds=round_idx, aborted=False)

            if not plan_text:
                # Agent did not submit a plan and did not stop → guard treats
                # this as "cannot proceed"; keep the last plan or force stop.
                if self._scratchpad.plan is None:
                    return self._finish(
                        rounds=round_idx,
                        aborted=True,
                        note="agent neither planned nor stopped",
                    )
                plan_text = self._scratchpad.plan

            self._scratchpad.submit_plan(plan_text, round_idx=round_idx)

            # 2. g: plan + PPF -> [StepSpec] (HTN solver via the planner).
            pipeline = self._build_pipeline(plan_text, ppf)
            if pipeline is None:
                # No plan from g — the agent must re-plan; record an empty
                # round and continue (bounded by max_rounds).
                self._scratchpad.record_round(
                    RoundRecord(round_idx=round_idx, goal_status={"planner": "no plan"})
                )
                continue

            # 3. f: execute.
            result = self._session.run(pipeline, problem)
            self._carry_pipeline_artifacts(round_idx, result)

            # 4. Evaluate: framework fact layer (objective, reproducible).
            goal_status = self._session.evaluate(result, goal_facts)

            # 5. Framework writes the round's actual (read-only trace).
            self._scratchpad.record_round(
                RoundRecord(
                    round_idx=round_idx,
                    goal_status={
                        "sat_facts": list(goal_status.sat_facts),
                        "unsat_facts": list(goal_status.unsat_facts),
                    },
                    artifacts=list(result.step_results.keys()),
                )
            )

            logger.info(
                "Round %d: steps=%d, sat=%s, unsat=%s",
                round_idx + 1,
                len(result.step_results),
                goal_status.sat_facts,
                goal_status.unsat_facts,
            )

        # Safety net: max_rounds exceeded → forced stop (not the agent's call).
        return self._finish(rounds=self._config.max_rounds, aborted=True)

    # -------------------------------------------------------------- helpers
    def _carry_pipeline_artifacts(self, round_idx: int, result: Any) -> None:
        """Carry every artifact produced by this round's pipeline into the
        cross-round handoff (the ``get_artifact`` resolution layer)."""
        from quro.core.cross_round_handoff import carry_result_artifacts

        carry_result_artifacts(self._handoff, round_idx, result)

    def _plan_round(self, round_idx: int) -> tuple[str | None, bool]:
        """Ask the agent to submit a plan (or stop) via the structured gate.

        Returns ``(plan_text, stop_requested)``.  The agent's decision is read
        from a fresh per-round ``PlanGate`` written by the ``submit_plan`` /
        ``stop`` tools — never from free-form response text.
        """
        from quro.planner.meta_tools import META_PLANNER_SYSTEM_PROMPT
        from quro.core.tool_surface import make_tool_surface

        controller_gate = ControllerPlanGate()
        builder = StepBuilder(
            step_types=self._step_types,
            handoff=self._handoff,
            audit_log=self._audit_log,
        )
        controller = RoundController(
            round_idx=round_idx,
            builder=builder,
            controller=self._session,
            gate=controller_gate,
            audit_log=self._audit_log,
            handoff=self._handoff,
        )
        controller.open()

        catalog_view = StepCatalogAdapter(self._step_types)
        tools = make_tool_surface(
            round_controller=controller,
            catalog_view=catalog_view,
            handoff=self._handoff,
            gate=controller_gate,
        )

        context = self._build_plan_context(round_idx)

        try:
            self._backend.complete(
                context,
                system=META_PLANNER_SYSTEM_PROMPT,
                tools=tools,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MetaPlanner round planning failed")
            controller.close()
            return None, False

        controller.close()

        # Read the structured gate — the agent's explicit decision.
        if controller_gate.stop:
            if controller_gate.conclusion:
                self._scratchpad.conclude(controller_gate.conclusion)
            return None, True
        if controller_gate.plan_text is not None and controller_gate.plan_text.strip():
            return controller_gate.plan_text.strip(), False
        # Neither submitted nor stopped → the round cannot proceed.
        return None, False

    def _build_pipeline(self, plan_text: str, ppf: PPF) -> Any | None:
        """g: run the HTN solver to turn the plan + PPF into a Pipeline.

        Returns ``None`` when the solver cannot produce a plan.
        """
        ppf_note = "\n".join(
            [f"Problem: {ppf.problem}"]
            + [f"Goal: {g}" for g in ppf.goals]
            + [f"Constraint: {c}" for c in ppf.constraints]
            + (
                ["Exploration skeleton (branches):"]
                + [f"- {b}" for b in ppf.plan]
                if ppf.plan
                else []
            )
        )
        # The plan is the agent's own draft, framed in echo voice so the
        # execution layer treats it as revisable intent, not an instruction
        # to execute verbatim (meta-plan-prompt-and-ppf.md §4.3).
        bound_problem = (
            "The agent's current plan (revisable):\n"
            f"{plan_text}\n\n"
            "Problem description (PPF):\n"
            f"{ppf_note}"
        )

        self._planner.init(
            bound_problem,
            self._step_types,
            goal_facts=[],
        )
        pef = self._planner.solve()
        if not pef:
            logger.warning("HTN solver returned empty plan")
            return None
        return self._planner.pef_to_pipeline(
            pef, default_config=self._pipeline_config,
        )

    def _build_plan_context(self, round_idx: int) -> str:
        """Assemble the agent's per-round context using structured blocks.

        Returns a flat string for the ``context`` parameter of
        ``backend.complete()``, but internally built from composable
        ``PromptBlock`` instances via ``MetaPlannerPromptManager``.
        """
        from quro.planner.meta_prompt import MetaPlannerPromptManager

        manager = MetaPlannerPromptManager()
        ppf = self._scratchpad.ppf

        extra_blocks = []
        if ppf:
            extra_blocks.append(manager.build_ppf_problem_block(ppf.problem))
            extra_blocks.append(manager.build_ppf_goals_block(ppf.goals))
            if ppf.constraints:
                extra_blocks.append(manager.build_ppf_constraints_block(ppf.constraints))
            extra_blocks.append(manager.build_ppf_plan_skeleton_block(ppf.plan))

        # Rounds trace (weighted, trimmable).
        rounds = self._scratchpad.rounds()
        if rounds:
            round_dicts = [
                {
                    "round_idx": r.round_idx,
                    "sat_facts": r.goal_status.get("sat_facts", []),
                    "unsat_facts": r.goal_status.get("unsat_facts", []),
                }
                for r in rounds
            ]
            extra_blocks.append(manager.build_rounds_trace_block(round_dicts))

        # Current round context (required).
        extra_blocks.append(
            manager.build_round_context_block(
                round_idx,
                plan_text=self._scratchpad.plan,
            )
        )

        # Assemble only the context blocks (no system prompt — that's
        # passed separately via ``system=META_PLANNER_SYSTEM_PROMPT``).
        ordered = manager._order(extra_blocks)
        return manager._render(ordered)

    def _finish(self, *, rounds: int, aborted: bool, note: str = "") -> MetaPlannerLoopResult:
        """Assemble the terminal result (report + goal_status + conclusion)."""
        ppf = self._scratchpad.ppf
        rounds_list = self._scratchpad.rounds()
        last_goal_status: dict[str, Any] = {}
        if rounds_list:
            last_goal_status = rounds_list[-1].goal_status
        return MetaPlannerLoopResult(
            rounds=rounds,
            report="" if aborted else self._scratchpad.conclusion or "",
            goal_status=last_goal_status,
            conclusion=self._scratchpad.conclusion,
            aborted=aborted,
        )

    # ------------------------------------------------------------ no-op mem
    @staticmethod
    def _noop_bridge() -> Any:
        from quro.memory.bridge import NoopMemoryBridge

        return NoopMemoryBridge()


class _NoopStore:
    """Minimal no-op resource store for conformance wiring check."""

    pass
