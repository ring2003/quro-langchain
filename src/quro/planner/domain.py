"""PlannerDomain — bridges quro-thinking's ExecutionPlanningDomain
into quro-langchain's pipeline model.

Maps:
  StepType           → PrimitiveOperator  (register_operator)
  Method (task→subtasks) → decomposition rule (register_method)
  PEF ExecutionPlan  → Pipeline + StepSpecs (pef_to_pipeline)
"""

from __future__ import annotations

import logging
from typing import Any

from quro.core.domain.step_type import StepTypeCatalog
from quro.steps.core import StepSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PlannerDomain
# ---------------------------------------------------------------------------


class PlannerDomain:
    """Wraps quro-thinking's ``ExecutionPlanningDomain`` to serve as the
    dynamic planning engine for quro-langchain pipelines.

    The planner maintains a single ``ReasoningSession`` across the policy
    loop.  After each pipeline run the step results are fed back, and
    ``solve()`` produces an updated PEF ready to be converted into a
    new ``Pipeline``.

    Args:
        backend: Optional LLM backend for planning calls (not yet used;
            the HTN solver is symbolic).  Kept for future LLM-augmented
            planning.
        max_reductions: Passed to ``solve_htn`` as the reduction limit.
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        max_reductions: int = 100,
        step_types: Any = None,
        skills: Any = None,
    ) -> None:
        self._backend = backend
        self._max_reductions = max_reductions
        self._session: Any = None
        self._step_types: StepTypeCatalog | None = step_types
        self._custom_methods: list[dict[str, Any]] = []
        # Phase D: optional domain catalogs injected into the planner context.
        self._skills = skills

    # ------------------------------------------------------------------ init
    def init(
        self,
        problem: str,
        step_types: StepTypeCatalog,
        *,
        goal_facts: list[str] | None = None,
        method_definitions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the planning session with *problem* and *step_types*.

        Registers every StepType as a ``PrimitiveOperator`` so the HTN solver
        can schedule them.  Generates a default method that decomposes
        ``solve_problem`` into all step types as sequential subtasks.  Custom
        method definitions (e.g. from ``register_method``) are also
        registered.

        Args:
            problem: The problem statement / objective.
            step_types: The StepType catalog available for execution.
            goal_facts: Fact names that must be ``True`` for the problem to
                be considered solved.  Passed through to the HTN solver as
                ``problem.goal_facts``.  If None or empty, goal validation
                is handled by the pipeline evaluate stage only.
            method_definitions: Optional pre-defined HTN decomposition
                methods.

        After calling this, ``solve()`` will produce a PEF plan.
        """
        from quro_thinking.domains.execution_planning import ExecutionPlanningDomain
        from quro_thinking.kernel import ReasoningSession

        self._step_types = step_types

        domain = ExecutionPlanningDomain()
        self._session = ReasoningSession(domain)

        operators = self._step_types_to_operators(step_types)
        methods = list(method_definitions or [])

        # Generate a default method that chains all step types sequentially.
        if not any(m.get("task_name") == "solve_problem" for m in methods):
            methods.append(self._build_sequential_method(operators))

        init_resp = self._session.call(
            "ep_init",
            {
                "problem": {
                    "objective": problem,
                    "goal_task": "solve_problem",
                    "goal_task_params": {},
                    "initial_facts": {},
                    "hard_constraints": [],
                    "goal_facts": list(goal_facts or []),
                    "capabilities": list(step_types.names()),
                    "actors": [],
                },
                "operators": operators,
                "methods": methods,
            },
        )
        if not init_resp.ok:
            raise RuntimeError(f"PlannerDomain init failed: {init_resp.error}")

    # ------------------------------------------------ step type → operator map
    @staticmethod
    def _step_types_to_operators(step_types: StepTypeCatalog) -> list[dict[str, Any]]:
        """Convert each ``StepType`` to a ``PrimitiveOperator`` for the HTN solver."""
        operators: list[dict[str, Any]] = []
        for name in step_types.names():
            step_type = step_types.get(name)
            if step_type is None:
                continue
            description = step_type.identity_block or name
            operators.append({
                "name": name,
                "description": description,
                "parameters": {},
                "preconditions": [],
                "effects": [
                    {"fact": f"{name}_done", "op": "set", "value": True},
                ],
                "cost": 1.0,
            })
        return operators

    @staticmethod
    def _build_sequential_method(
        operators: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a default HTN method that chains all operators sequentially.

        Subtask ``i`` depends on subtask ``i-1`` via ordering constraints.
        """
        subtasks: list[dict[str, Any]] = []
        ordering: list[list[int]] = []
        for i, op in enumerate(operators):
            subtasks.append({
                "name": op["name"],
                "kind": "primitive",
                "parameters": {},
                "order_index": i,
            })
            if i > 0:
                ordering.append([i - 1, i])

        return {
            "id": "solve_problem_sequential",
            "task_name": "solve_problem",
            "preconditions": [],
            "subtasks": subtasks,
            "ordering": ordering,
            "cost": sum(op.get("cost", 1.0) for op in operators),
            "description": "Sequential execution of all step types.",
            "tags": ["sequential"],
        }

    # -------------------------------------------------------------- register
    def register_method(self, method: dict[str, Any]) -> bool:
        """Register a decomposition method in the planning domain.

        Args:
            method: A dict with keys matching the HTN ``Method`` schema:
                id, task_name, subtasks, ordering, cost, description, tags.
        """
        if self._session is None:
            return False
        resp = self._session.call("ep_register_method", {"method": method})
        if resp.ok:
            self._custom_methods.append(dict(method))
        return resp.ok

    def register_operator(self, operator: dict[str, Any]) -> bool:
        """Register a new primitive operator (e.g. for a dynamic step type)."""
        if self._session is None:
            return False
        resp = self._session.call("ep_register_operator", {"operator": operator})
        return resp.ok

    def get_custom_methods(self) -> list[dict[str, Any]]:
        """Return methods registered via ``register_method`` since construction.

        Only includes methods that were added after ``init()`` (i.e. by
        MetaPlanner's ``define_method`` tool).  Does NOT include methods
        passed via ``init(method_definitions=...)`` or the auto-generated
        default sequential method.

        Returns:
            List of method dicts (deep-copied).
        """
        return [dict(m) for m in self._custom_methods]

    @property
    def session(self) -> Any:
        """Expose the underlying ``ReasoningSession`` for state inspection."""
        return self._session

    # ---------------------------------------------------------------- solve
    def solve(self) -> dict[str, Any]:
        """Run the HTN solver and return the PEF mid-artifact.

        Returns:
            The PEF dict, or an empty dict on failure.
        """
        if self._session is None:
            logger.error("PlannerDomain.solve() called before init()")
            return {}

        solve_resp = self._session.call(
            "ep_solve", {"max_reductions": self._max_reductions}
        )
        if not solve_resp.ok:
            logger.warning("HTN solver failed: %s", solve_resp.error)
            return {}

        pef_resp = self._session.call("ep_export_pef", {})
        if not pef_resp.ok:
            logger.warning("PEF export failed: %s", pef_resp.error)
            return {}

        pef = pef_resp.extra.get("pef", {}) if pef_resp.extra else {}
        return pef

    # ------------------------------------------------------ PEF → Pipeline
    def pef_to_pipeline(
        self,
        pef: dict[str, Any],
        *,
        default_config: Any = None,
    ) -> Any:
        """Convert a PEF mid-artifact to a ``Pipeline`` of ``StepSpec``s.

        This is the outer planner's ``g : Problem → [StepSpec]``: each PEF task
        becomes exactly one executor ``StepSpec`` (one step type, one task),
        with causal dependencies resolved to ``depends_on`` step ids.  There is
        no ``plan`` generator step and no expander — a step is a step; the
        planner emits ``[StepSpec]`` directly (AGENTS.md "g/f separation").

        Args:
            pef: The PEF dict from ``solve()``.
            default_config: Optional ``PipelineConfig`` override.

        Returns:
            A ``Pipeline``.
        """
        from quro.pipeline.core import Pipeline, PipelineConfig

        exec_graph = pef.get("executionGraph", {})
        tasks = exec_graph.get("tasks", [])
        causal = pef.get("causalDependencies", [])

        # Build index: PEF task id → task dict.
        task_index: dict[str, dict[str, Any]] = {t["id"]: t for t in tasks}
        step_types = self._step_types

        steps: list[StepSpec] = []
        for task in tasks:
            task_id = task["id"]
            task_name = task.get("name", "unknown")
            # Resolve dependency edges: every task this task depends on.
            pred_ids: list[str] = sorted({
                edge["from"]
                for edge in causal
                if edge.get("to") == task_id and edge.get("from") in task_index
            })
            # Also inherit from HTN predecessors (embedded in task dict).
            for pred in task.get("predecessors", []):
                if pred in task_index and pred not in pred_ids:
                    pred_ids.append(pred)

            step_type = step_types.get(task_name) if step_types else None
            steps.append(StepSpec(
                id=task_id,
                name=task_name,
                step_type=task_name,
                objective=f"Step '{task_name}': execute your assigned task.",
                depends_on=pred_ids,
                # Carry the StepType's tool grant (capability markers such as
                # Readonly / Shell / Write) onto the StepSpec so the
                # ToolCoordinator can resolve the explorer's external tools
                # (read/ls/grep/find/shell).  Without this the planner emits
                # empty `tools=[]` and every executor step is toolless.  Empty
                # (not None) — the coordinator iterates `step.tools`.
                tools=(getattr(step_type, "tools", None) or ()),
                features=step_types.features_for(task_name) if step_types else (),
                mounted=step_types.mounted_for(task_name) if step_types else None,
            ))

        return Pipeline(
            name=pef.get("problemId", "planned")[:60],
            steps=steps,
            config=default_config or PipelineConfig(),
        )

    # ------------------------------------------------------- feedback loop
    def update_state(
        self,
        completed_task_ids: list[str],
        new_facts: dict[str, Any] | None = None,
    ) -> None:
        """Tell the planner which tasks have completed.

        After pipeline steps execute, call this to mark task IDs as done
        before calling ``solve()`` again to replan remaining work.
        """
        if self._session is None:
            return
        state = self._session.state or {}
        network_data = state.get("task_network") or {}
        tasks = dict(network_data.get("tasks") or {})

        for tid in completed_task_ids:
            if tid in tasks:
                tasks[tid] = {**tasks[tid], "status": "completed"}

        network_data["tasks"] = tasks
        new_state = {**state, "task_network": network_data}

        # Consume new_facts into the solver's fact base so the next solve()
        # sees them (e.g. ``{name}_unsat: True`` facts to re-plan around).
        if new_facts:
            exec_state = dict(new_state.get("execution_state") or {})
            facts = dict(exec_state.get("facts") or {})
            facts.update(dict(new_facts))
            exec_state["facts"] = facts
            new_state["execution_state"] = exec_state

        self._session._state = new_state

    # ----------------------------------------------------------- properties
    @property
    def session(self) -> Any:
        """The underlying quro-thinking ``ReasoningSession``."""
        return self._session

    @property
    def step_types(self) -> StepTypeCatalog | None:
        """The StepType catalog used by this planner."""
        return self._step_types
