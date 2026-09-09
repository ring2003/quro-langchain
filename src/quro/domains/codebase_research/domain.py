"""Codebase research domain — user-authored domain extension (Phase D).

The domain supplies the *step-type catalog* for progressive codebase
exploration — phase vocabulary, StepTypes, skills, roles, and the round-object
state shape — while the **MetaPlanner keeps the exploration control flow**
(narrowing decisions).  See
``docs/architecture/codebase-research-domain.md``.

The domain is self-contained: it inherits quro-thinking's ``Domain`` ABC
directly (not ``EngineeringWorkflowDomain``), so it owns its phase set, its
domain ops, and its state shape without coupling to the engineering domain's
implementation details.
"""

from __future__ import annotations

from uuid import uuid4
from typing import Any

from quro_thinking.kernel import Domain, DomainState, ToolResponse, VerifyResult

from quro.context.blocks import phase_blocks
from quro.core.domain.step_type import StepType, StepTypeCatalog
from quro.core.features import Feature
from quro.core.tools.capability import Readonly, Shell, Write
from quro.skills import SkillCatalog

# Phase vocabulary (the skeleton).  Phase *progression* is a MetaPlanner
# decision — the domain never auto-advances between phases (blueprint §2).
RESEARCH_PHASES: tuple[str, ...] = (
    "survey",           # 总览结构 — overview the repository structure
    "narrow",           # 定位相关模块 — locate the relevant modules
    "deep_dive",        # 读关键文件 — read the key files
    "synthesize",       # 跨模块关联 — relate findings across modules
    "assemble_report",  # 组装报告 — merge chunks into the final report
)

# StepType → default phase.  The runtime derives a step's session phase from
# its step type, so each step runs with the phase gating that matches its
# cognitive mode (blueprint §3.2).
STEP_TYPE_PHASE: dict[str, str] = {
    "survey_module": "survey",
    "deep_dive_symbol": "deep_dive",
    "assemble_report": "assemble_report",
}

# Runtime-internal state-handoff op (StepAdapter calls it before governor
# execution to carry structured dependency data into this session's state —
# fix-v20260906.md).  Allowed in every phase; not a model tool.
_CARRY_FORWARD = "carry_forward"

# Ops shared by every exploration phase.
_EXPLORE_OPS: frozenset[str] = frozenset({
    "commit_clue",
    "continue_step",
    "add_artifact",
    "complete_step",
    "terminate_subtree",
    _CARRY_FORWARD,
})

# Ops allowed only when assembling the report.
_REPORT_OPS: frozenset[str] = frozenset({
    "assemble_report",
    "add_artifact",
    "complete_step",
    "terminate_subtree",
    _CARRY_FORWARD,
})

# phase → allowed domain ops (single gating authority for this domain).
CODEBASE_RESEARCH_PHASE_OPS: dict[str, frozenset[str]] = {
    "survey": _EXPLORE_OPS,
    "narrow": _EXPLORE_OPS,
    "deep_dive": _EXPLORE_OPS,
    "synthesize": _EXPLORE_OPS,
    "assemble_report": _REPORT_OPS,
}

# Ops valid in every phase (kernel-intercepted: checkpoint/backtrack).
ALLOWED_IN_ANY_PHASE: frozenset[str] = frozenset({"checkpoint", "backtrack"})


def allowed_ops_for_phase(phase: str) -> set[str]:
    """Return the ops allowed in *phase*, including any-phase ops."""
    return set(CODEBASE_RESEARCH_PHASE_OPS.get(phase, set())) | set(ALLOWED_IN_ANY_PHASE)


def default_step_types() -> StepTypeCatalog:
    """Return the codebase-research StepType catalog (blueprint §3.2)."""
    return StepTypeCatalog([
        StepType(
            "survey_module",
            skill_pool=["codegraph", "read-strategy"],
            identity_block=(
                "You are the **survey_module** step. Your responsibility: "
                "overview the repository structure and record the relevant "
                "modules. Focus ONLY on your assigned step.\n\n"
                "## CONSTRAINTS \n"
                "- commit_clue() ALWAYS on every step!"
                "- add_artifact() at last."
                "- complete_step() as finalize."
            ),
            executor_hints={"phase": "survey", "terminal_tool": "complete_step"},
            features=(Feature.RECOVERY,),
            tools=(Readonly, Shell),
            access="hints",
        ),
        StepType(
            "deep_dive_symbol",
            skill_pool=["codegraph", "evidence-format"],
            identity_block=(
                "You are the **deep_dive_symbol** step. Your responsibility: "
                "read the key files and record evidence with commit_clue whenever you have any progress. Focus ONLY on your "
                "assigned step. \n\n"
                "## CONSTRAINTS \n"
                "- commit_clue() ALWAYS on every step!"
                "- add_artifact() at last."
                "- complete_step() as finalize."
            ),
            executor_hints={"phase": "deep_dive", "terminal_tool": "complete_step"},
            features=(Feature.RECOVERY,),
            tools=(Readonly, Shell),
            access="hints",
        ),
        StepType(
            "assemble_report",
            skill_pool=["report-template"],
            identity_block=(
                "You are the **assemble_report** step. Your responsibility: "
                "merge findings into the final report. Focus ONLY on your "
                "assigned step."
            ),
            executor_hints={"phase": "assemble_report", "terminal_tool": "complete_step"},
            tools=(Readonly, Write),
            access="hints",
        ),
    ])


def default_skills() -> SkillCatalog:
    """Load the domain's skill resources from ``skills/`` (Anthropic ``.md``).

    Falls back to an empty catalog when the package resources are not
    available (e.g. a source tree without the ``.md`` files installed).
    """
    try:
        from importlib import resources

        return SkillCatalog.from_dir(
            resources.files("quro.domains.codebase_research") / "skills"
        )
    except Exception:  # pragma: no cover - defensive
        return SkillCatalog()


@phase_blocks({
    "survey": ["problem", "plan", "own_history", "artifacts", "module_inventory", "recovery_clues", "objective"],
    "narrow": ["problem", "plan", "module_inventory", "hints", "own_history", "recovery_clues", "objective"],
    "deep_dive": ["problem", "plan", "hints", "own_history", "recovery_clues", "objective"],
    "synthesize": ["problem", "plan", "evidence", "own_history", "recovery_clues", "objective"],
    "assemble_report": ["problem", "evidence", "report_chunks", "objective"],
})
class CodebaseResearchDomain(Domain):
    """A domain extension for progressive codebase exploration + reporting.

    Declares the research phase skeleton, its StepTypes and skills, and the
    round-object state shape.  The domain does **not** own phase progression —
    the MetaPlanner enters each phase via ``build_pipeline(phase=…)`` and the
    phase lands in ``initial_state`` through the init args.
    """

    name: str = "codebase_research"

    # InlineStep recovery declaration (primitive-step.md §6): the domain
    # declares its recovery need; the upper layer (demo launcher / resume
    # entry) reads it and passes it to the RecoveryCoordinator — the
    # coordinator does not re-derive it from the domain.  Same declaration
    # shape as ``EngineeringWorkflowDomain``.
    #
    # Default: enable compaction inlining for the *explorer* step types only
    # (survey_module / deep_dive_symbol) via ``backtrack_plan``; the reporter
    # step (assemble_report) keeps plain structural resume.
    recovery_mode: str = "compaction"  # "none" | "compaction" | "mount"
    recovery_compaction: str | None = None  # fallback when mode == "compaction"
    recovery_compaction_for_step_type: dict[str, str] = {
        "survey_module": "backtrack_plan",
        "deep_dive_symbol": "backtrack_plan",
    }

    # Steering declaration (primitive-step.md §6/§7): the explorer step types
    # open the invasive objective-override loop (each act narrows to the next
    # objective).  The ``narrow`` steering act + identity are declared here; a
    # small retry_budget fits exploration (an unreadable symbol -> move on).
    steering_for_step_type: dict[str, str] = {
        "survey_module": "narrow",
        "deep_dive_symbol": "narrow",
    }
    steering_retry_budget: int = 1  # exploration: hand to steering quickly
    # TEMP-DEBUG: steering round budget raised to 999 so a step is not cut short by
    # the budget before it reaches complete_step; revert to 8 once debugged.
    steering_max_rounds: int = 999   # max objective overrides per step

    # Steering gate and folder (primitive-step.md 3.2/4): protocol-driven
    # completion and folding logic for the SteeringLoop.
    steering_gate: Any = None   # set in __init__
    steering_folder: Any = None  # set in __init__

    def __init__(
        self,
        *,
        step_types: StepTypeCatalog | None = None,
        skills: SkillCatalog | None = None,
        recovery_mode: str | None = None,
        recovery_compaction: str | None = None,
        recovery_compaction_for_step_type: dict[str, str] | None = None,
        steering_for_step_type: dict[str, str] | None = None,
        steering_retry_budget: int | None = None,
        steering_max_rounds: int | None = None,
    ) -> None:
        super().__init__()
        self._step_types = step_types or default_step_types()
        self._skills = skills or default_skills()
        if recovery_mode is not None:
            self.recovery_mode = recovery_mode
        if recovery_compaction is not None:
            self.recovery_compaction = recovery_compaction
        if recovery_compaction_for_step_type is not None:
            self.recovery_compaction_for_step_type = dict(
                recovery_compaction_for_step_type
            )
        if steering_for_step_type is not None:
            self.steering_for_step_type = dict(steering_for_step_type)
        if steering_retry_budget is not None:
            self.steering_retry_budget = steering_retry_budget
        if steering_max_rounds is not None:
            self.steering_max_rounds = steering_max_rounds

        # Wire up steering gate and folder.
        self.steering_gate = _CodebaseResearchSteeringGate(
            max_rounds=self.steering_max_rounds
        )
        self.steering_folder = _CodebaseResearchSteeringFolder()

    # -- step-type catalog -------------------------------------------------

    @property
    def step_types(self) -> StepTypeCatalog:
        """The domain's StepType catalog (planner-context injection)."""
        return self._step_types

    @property
    def skills(self) -> SkillCatalog:
        """The domain's skill catalog (metadata + on-demand bodies)."""
        return self._skills

    def get_skill_metadata(self) -> list[dict[str, Any]]:
        """Return ``[{name, path, description}, ...]`` (quro-thinking FR-3)."""
        return self._skills.metadata()

    def phases(self) -> list[str]:
        """Return the declared phase vocabulary (for MetaPlanner injection)."""
        return list(RESEARCH_PHASES)

    def phase_for_step_type(self, step_type: str) -> str:
        """Return the default phase for *step_type* (survey when unknown)."""
        return STEP_TYPE_PHASE.get(step_type, "survey")

    def allowed_ops_for_phase(self, phase: str) -> set[str]:
        """Return the ops allowed in *phase* (instance-level, for the runtime)."""
        return allowed_ops_for_phase(phase)

    # -- lifecycle ---------------------------------------------------------

    def initial_state(self, init_args: dict[str, Any]) -> DomainState:
        phase = str(init_args.get("phase") or "survey")
        if phase not in CODEBASE_RESEARCH_PHASE_OPS:
            phase = "survey"
        return {
            "phase": phase,
            "problem": init_args.get("problem", ""),
            # decision anchor (pre-step intent) for overrun-prone steps.
            "exploration_plan": "",
            # module inventory: [{module, path, relevance}]
            "module_inventory": [],
            # per-phase report-chunk file paths.
            "report_chunks": [],
            # final report file path (set by assemble_report).
            "report_path": "",
            "steps": [],
            "executing_step_id": None,
            "artifacts": [],
            # step-scoped recovery journal (filled by Stage 3):
            # step_id -> {step_round, clues[], continue_reasons[], failed_reasons[]}
            "recovery": {},
        }

    def apply(
        self, state: DomainState, op: str, args: dict[str, Any]
    ) -> tuple[DomainState, ToolResponse]:
        data = dict(state)

        phase = data.get("phase", "survey")
        allowed = allowed_ops_for_phase(phase)
        if op not in allowed:
            return data, ToolResponse(
                ok=False,
                error=f"PHASE_MISMATCH: '{op}' not allowed in phase '{phase}'",
                explanation=f"Allowed ops in phase '{phase}': {', '.join(sorted(allowed))}",
            )

        if op == "commit_clue":
            # Append-only, step-scoped, small-and-frequent (blueprint §6 stage 2).
            step_id = args.get("step_id") or _default_step_id(data)
            clue = str(args.get("clue", "")).strip()
            if not clue:
                return data, ToolResponse(
                    ok=False,
                    error="commit_clue requires 'clue'",
                    explanation="Pass clue=<one compact finding> to persist it.",
                )
            recovery = _copy_recovery(data.get("recovery"))
            journal = recovery.setdefault(step_id, _new_journal())
            journal["clues"] = journal.get("clues", []) + [
                {"clue_id": f"cl_{uuid4().hex[:6]}", "text": clue},
            ]
            recovery[step_id] = journal
            data["recovery"] = recovery

        elif op == "continue_step":
            # Record a recovery continue decision: resume exploration from the
            # existing clues (blueprint §6).  Bumps step_round.
            step_id = args.get("step_id") or _default_step_id(data)
            reason = str(args.get("reason", "")).strip()
            recovery = _copy_recovery(data.get("recovery"))
            journal = recovery.setdefault(step_id, _new_journal())
            if reason:
                journal["continue_reasons"] = journal.get("continue_reasons", []) + [reason]
            journal["step_round"] = journal.get("step_round", 0) + 1
            recovery[step_id] = journal
            data["recovery"] = recovery

        elif op == "assemble_report":
            path = str(args.get("path", ""))
            title = str(args.get("title", ""))
            chunks = list(data.get("report_chunks", []))
            if path and path not in chunks:
                chunks.append(path)
            data["report_chunks"] = chunks
            data["report_path"] = path or data.get("report_path", "")
            if title:
                data["report_title"] = title

        elif op == "add_artifact":
            artifacts = list(data.get("artifacts", []))
            artifacts.append({
                "artifact_id": args.get("artifact_id", f"art_{uuid4().hex[:8]}"),
                "step_id": args.get("step_id", _default_step_id(data)),
                "kind": args.get("kind", "analysis"),
                "summary": args.get("summary", ""),
                "evidences": args.get("evidences", []),
                "body": args.get("body", ""),
            })
            data["artifacts"] = artifacts

        elif op == "complete_step":
            step_id = args.get("step_id", "") or _default_step_id(data)
            if not step_id:
                return data, ToolResponse(
                    ok=False,
                    error="complete_step requires step_id",
                    explanation="Pass step_id=... to identify which step to complete.",
                )
            steps = list(data.get("steps", []))
            current_step = None
            for s in steps:
                if s["step_id"] == step_id:
                    current_step = s
                    break
            if current_step is None:
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' not found",
                    explanation="Use create_step / seed the step first.",
                )

            # NEW: Set round_status only — this is L0's sole authority.
            # step_status is NOT touched here; L1 (or L0 for ordinary steps) owns it.
            if current_step.get("round_status") == "completed":
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' round already completed",
                    explanation="This governor round is already complete.",
                )

            step_artifacts = [
                a for a in data.get("artifacts", [])
                if a.get("step_id") == step_id
            ]
            if not step_artifacts:
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' has no artifacts",
                    explanation="Call add_artifact before complete_step.",
                )

            # Determine whether this step is steering-controlled.  For steered
            # steps, step_status is the L1-owned token — round_status is the
            # only token complete_step may write; the fold in
            # StepAdapter._execute_with_steering mints step_status only when
            # the steering loop exited on a Steering-approved terminal
            # (final round declared, or budget exhausted with artifacts —
            # Steering-Semantics.md §12-15).  For ordinary steps L0 == L1,
            # so complete_step may mint both tokens.
            # Keying pipeline advancement on step_status keeps
            # executing_step_id set across every L0 round of a steered step
            # (the fold may hold the step open while it narrows).
            is_steered = bool(
                self.steering_for_step_type.get(
                    current_step.get("step_type", ""), ""
                )
            )
            for s in steps:
                if s["step_id"] == step_id:
                    s["round_status"] = "completed"
                    if not is_steered:
                        s["step_status"] = "completed"
                    break
            data["steps"] = steps
            remaining = [
                s for s in steps
                if s.get("step_status") not in ("completed",)
            ]
            if not remaining:
                data["executing_step_id"] = None
            else:
                data["executing_step_id"] = remaining[0]["step_id"]

        elif op == "terminate_subtree":
            step_id = args.get("step_id") or _default_step_id(data)
            steps = list(data.get("steps", []))
            for s in steps:
                if s["step_id"] == step_id:
                    s["round_status"] = "terminated"
                    break
            data["steps"] = steps

        elif op == "carry_forward":
            # State handoff between step sessions (mirrors
            # EngineeringWorkflowDomain.carry_forward).  The runtime calls this
            # before governor execution with ``sources=[{step_id, <key>: ...}]``
            # so structured dependency data reaches this session's state —
            # without it the hints block, steering_artifacts block, and
            # ContextView.dependencies render empty for research steps
            # (fix-v20260906.md).  Only ``steps`` and ``artifacts`` are carried:
            # they are the keys the prompt blocks and steering context view
            # resolve dependencies from.
            sources = args.get("sources", [])
            if isinstance(sources, list):
                for src in sources:
                    if not isinstance(src, dict):
                        continue
                    for key in ("steps", "artifacts"):
                        src_data = src.get(key)
                        if not isinstance(src_data, list):
                            continue
                        if key == "steps":
                            existing_ids = {
                                s.get("step_id") for s in data.get("steps", [])
                            }
                            for s in src_data:
                                sid = s.get("step_id")
                                if sid and sid not in existing_ids:
                                    data.setdefault("steps", []).append(s)
                                    existing_ids.add(sid)
                        else:  # artifacts — dedupe by artifact_id; the original
                            # step_id tag is what lets hints/steering filter deps.
                            existing_ids = {
                                a.get("artifact_id")
                                for a in data.get("artifacts", [])
                            }
                            for a in src_data:
                                aid = a.get("artifact_id")
                                if aid and aid not in existing_ids:
                                    data.setdefault("artifacts", []).append(a)
                                    existing_ids.add(aid)
            # NOTE: data["phase"] is intentionally NOT overwritten — the session
            # was initialised with the phase derived from THIS step's type, and
            # the adapter may pass a foreign target phase (e.g. "step_execute")
            # that is not a research phase; applying it would break gating.
            return data, ToolResponse(
                ok=True, explanation="Carried forward dependency state."
            )

        return data, ToolResponse(ok=True, operation={op: True})

    def verify(self, state: DomainState, scope: str = "all") -> VerifyResult:
        errors: list[str] = []
        if not state.get("artifacts"):
            errors.append("missing artifacts")
        if state.get("phase") == "assemble_report" and not state.get("report_path"):
            errors.append("missing report")
        return VerifyResult(satisfied=[], violated=errors, pending=[])

    def is_complete(self, state: DomainState) -> bool:
        """A round is complete once a report is recorded and artifacts exist."""
        return bool(state.get("report_path")) and bool(state.get("artifacts"))


def _default_step_id(data: dict[str, Any]) -> str:
    return str(data.get("executing_step_id") or "")


def _new_journal() -> dict[str, Any]:
    """Return a fresh step recovery journal (blueprint §6)."""
    return {
        "step_round": 0,
        "clues": [],
        "continue_reasons": [],
        "failed_reasons": [],
    }


def _copy_recovery(recovery: Any) -> dict[str, Any]:
    """Deep-copy the recovery journal map so ``apply`` stays pure."""
    out: dict[str, Any] = {}
    for step_id, journal in (recovery or {}).items():
        copied: dict[str, Any] = {}
        for key, value in (journal or {}).items():
            copied[key] = list(value) if isinstance(value, list) else value
        out[step_id] = copied
    return out


# ---------------------------------------------------------------------------
# Steering gate and folder (primitive-step.md 3.2/4)
# ---------------------------------------------------------------------------


class _CodebaseResearchSteeringGate:
    """Gate for the codebase-research steering loop (Steering-Semantics.md §16–17).

    suggested_sufficiency: provides a fact (artifact count) as a read-only hint.
        The gate does NOT make the completion decision. The steering act sees
        this in its prompt and decides frontier completion itself: it checks
        the frontier's done_when against the artifact evidence it receives and
        declares the final round (NextInstruction.is_final_round) when the
        investigation is complete.

    budget_exhausted: enforces the declared cycle cap (max_rounds) as a safety
        net. When True, the loop exits regardless of frontier status.

    Note: this gate carries no completion predicate. Frontier completion is the
    steering act's evidence-based judgment (its own done_when, checked against
    the steering_artifacts view); round_status — an L0-owned fact set by
    complete_step — never gates the loop's exit (Anti-pattern E).
    """

    def __init__(self, *, min_artifacts: int = 1, max_rounds: int = 8) -> None:
        self._min_artifacts = min_artifacts
        self._max_rounds = max_rounds

    def suggested_sufficiency(self, state: Any, *, objective: str) -> str:
        """Provide a read-only fact hint about frontier progress (Steering-Semantics.md §16).

        Returns a human-readable hint about artifact count. This is a FACT,
        not a decision — the steering act sees this in its prompt and weighs it
        against the frontier's done_when; the framework never uses it to exit
        the loop.
        """
        if state is None:
            return "No state available."
        artifacts = state.get("artifacts", []) if isinstance(state, dict) else []
        count = len(artifacts)
        ids = [a.get("artifact_id", "") for a in artifacts[:8]]
        id_str = ", ".join(ids) if ids else "(none)"
        return (
            f"{count} artifact(s) collected: [{id_str}]. "
            f"Domain default minimum is {self._min_artifacts}."
        )

    def budget_exhausted(self, state: Any, *, rounds: int) -> bool:
        # Honour the declared cap: stop the loop after max_rounds rounds
        # (previously hard-coded False — a steering step always burned the
        # full budget even when the work was done; F4).
        return rounds >= self._max_rounds


class _CodebaseResearchSteeringFolder:
    """Folder for the codebase-research steering loop.

    Merges artifacts from multiple steering rounds into one list.
    """

    def fold(self, state: Any, *, artifacts: list[Any]) -> Any:
        # Return the last non-None artifact as the terminal result.
        for a in reversed(artifacts):
            if a is not None:
                return a
        return None
