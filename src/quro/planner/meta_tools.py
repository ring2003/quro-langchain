"""MetaPlanner tool set for Phase 3 — LLM-assisted plan repair.

Tool summary:

================ ============================================ ===========
Tool              Wraps                                        Complexity
================ ============================================ ===========
create_step       Direct StepSpec creation (primary path)      low
run_custom_pipeline  Execute manually-created steps            medium
setup_step        Update a manual step's fields                low
list_manual_steps List manually-created steps                  low
define_method     PlannerDomain.register_method (HTN verify)  low
build_pipeline    solve() + pef_to_pipeline() (HTN verify)    low
run_pipeline      PipelineRunner.run() → summary (advisory)   medium
get_artifact      Retrieves artifact content from last run     low
recall_memory     IMemoryBridge.recall() / recall_patterns()  low
record_decision   IMemoryBridge.distill(type="decision")      low
commit_learning   IMemoryBridge.distill(type="procedural")    low
list_methods      Query registered HTN methods                 low
list_operators    Query registered HTN operators               low
================ ============================================ ===========
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool

from quro.core.domain.step_type import StepTypeCatalog
from quro.memory.protocols import IMemoryBridge
from quro.pipeline.core import Pipeline, PipelineConfig, PipelineResult, PipelineRunner
from quro.planner.domain import PlannerDomain
from quro.steps.core import StepSpec

logger = logging.getLogger(__name__)

# ============================================================================
# META_PLANNER_SYSTEM_PROMPT — imported from meta_prompt module
# ============================================================================

from quro.planner.meta_prompt import META_PLANNER_SYSTEM_PROMPT  # noqa: F811



# ============================================================================
# HTN Schema Helpers
# ============================================================================

def _build_operator_dict(name: str, description: str, cost: float = 1.0) -> dict[str, Any]:
    """Build a primitive operator dict matching quro-thinking HTN schema."""
    return {
        "name": name,
        "description": description,
        "parameters": {},
        "preconditions": [],
        "effects": [
            {"fact": f"{name}_done", "op": "set", "value": True},
        ],
        "cost": cost,
    }


def _parse_subtasks_json(raw: str) -> list[dict[str, Any]]:
    """Parse a subtasks JSON string into a list of subtask dicts.

    Validates required fields (name, kind).
    """
    try:
        subtasks = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for subtasks: {e}") from e

    if not isinstance(subtasks, list):
        raise ValueError("subtasks_json must be a JSON array")

    for i, st in enumerate(subtasks):
        if not isinstance(st, dict):
            raise ValueError(f"subtask[{i}] must be a JSON object")
        if "name" not in st:
            raise ValueError(f"subtask[{i}] missing required field 'name'")
        st.setdefault("kind", "primitive")
        st.setdefault("parameters", {})
        st.setdefault("order_index", i)

    return subtasks


def _build_method_dict(
    task_name: str,
    subtasks: list[dict[str, Any]],
    description: str = "",
    method_id: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete HTN method dict from subtasks.

    Auto-generates ordering constraints (sequential chain) and cost.
    """
    # Auto-generate ordering: sequential chain subtask[i-1] → subtask[i].
    ordering: list[list[int]] = []
    for i in range(1, len(subtasks)):
        ordering.append([i - 1, i])

    total_cost = sum(st.get("cost", 1.0) for st in subtasks)

    if not method_id:
        # Generate a stable id from task_name + subtask names.
        subtask_names = "_".join(st["name"] for st in subtasks)
        method_id = f"{task_name}_{subtask_names}"

    return {
        "id": method_id,
        "task_name": task_name,
        "preconditions": [],
        "subtasks": subtasks,
        "ordering": ordering,
        "cost": total_cost,
        "description": description or f"Decompose {task_name} into {len(subtasks)} subtasks.",
        "tags": tags or [],
    }


# ============================================================================
# Pipeline run summary helpers
# ============================================================================

def _summarize_pipeline_result(
    result: PipelineResult,
    goal_facts: list[str] | None = None,
    round_goal_facts: list[str] | None = None,
    goal_status_info: dict[str, Any] | None = None,
) -> str:
    """Summarize a PipelineResult as a self-contained report for MetaPlanner.

    The report includes step summaries, automatic goal evaluation (via
    ``goal_status_from_results``), and a goal-status summary.
    MetaPlanner only needs to read the report and decide — no need to
    "remember intent" across the tool call boundary.

    Report structure::

        ### Step Summary (N/N OK)
          - step_id: OK/FAILED — artifact description [art_xxx]
        ### Goal Status
          - goal_name: SAT/UNSAT — evidence
        ### Goal Status Summary: all-met | some-unmet | none-met
    """
    # Auto-evaluate goals against pipeline results.
    from quro.planner.goal_status import goal_status_from_results

    diagnosis = None
    if goal_facts:
        try:
            diagnosis = goal_status_from_results(result, goal_facts)
        except Exception:
            logger.warning("goal_status_from_results failed", exc_info=True)

    lines: list[str] = []
    lines.append("## Pipeline Execution Report")

    # -- Step Summary --------------------------------------------------------
    step_results = result.step_results
    if not isinstance(step_results, dict):
        if isinstance(step_results, (list, tuple)):
            step_results = {getattr(r, "step_id", f"step_{i}"): r
                            for i, r in enumerate(step_results)}
        else:
            step_results = {}

    total = len(step_results)
    ok_count = sum(1 for r in step_results.values() if r.ok)
    failed_count = total - ok_count

    lines.append(f"### Step Summary ({ok_count}/{total} OK)")
    if failed_count:
        lines.append(f"  ({failed_count} failed)")

    step_order = result.order or list(step_results.keys())
    for step_id in step_order:
        r = step_results.get(step_id)
        if r is None:
            lines.append(f"  - {step_id}: MISSING")
            continue

        if r.ok:
            # Extract artifact summary.
            artifacts = getattr(r, "artifacts", []) or []
            summary_text = ""
            art_ids: list[str] = []
            for a in artifacts:
                if isinstance(a, dict):
                    aid = a.get("artifact_id", "")
                    if aid:
                        art_ids.append(aid)
                    if not summary_text:
                        summary_text = str(a.get("summary", ""))[:120]
            if not summary_text:
                metrics = getattr(r, "metrics", {}) or {}
                summary_text = metrics.get("summary", "(no summary)")
            if not summary_text or summary_text == "(no summary)":
                summary_text = "(OK — no artifact summary)"

            art_str = f" [{', '.join(art_ids)}]" if art_ids else ""
            lines.append(f"  - {step_id}: OK — {summary_text}{art_str}")
        else:
            error_msg = getattr(r, "error", "unknown error") or "unknown error"
            lines.append(f"  - {step_id}: FAILED — {error_msg[:200]}")

    # -- Round Goal Status (artifact/file acceptance) -------------------------
    round_round_facts = list(round_goal_facts or [])
    round_sat: list[str] = []
    round_unsat: list[str] = []
    if round_round_facts:
        from quro.planner.goal_status import FileArtifactChecker
        checker = FileArtifactChecker()
        round_sat, round_unsat = checker.check(round_round_facts, result)
        lines.append("")
        lines.append("### Round Goal Status")
        for rgf in round_round_facts:
            if rgf in round_sat:
                lines.append(f"  - {rgf}: SAT")
            else:
                lines.append(f"  - {rgf}: UNSAT")

    # -- Final Goal Status ----------------------------------------------------
    lines.append("")
    lines.append("### Final Goal Status")

    if diagnosis is not None:
        sat_facts = diagnosis.sat_facts
        unsat_facts = diagnosis.unsat_facts
        for gf in (goal_facts or []):
            if gf in sat_facts:
                lines.append(f"  - {gf}: SAT")
            elif gf in unsat_facts:
                evidence = ""
                if diagnosis.cause:
                    evidence = f" — {diagnosis.cause[:100]}"
                lines.append(f"  - {gf}: UNSAT{evidence}")
            else:
                lines.append(f"  - {gf}: UNKNOWN")
        # Also show any extra facts from diagnosis not in goal_facts.
        extra_sat = set(sat_facts) - set(goal_facts or [])
        extra_unsat = set(unsat_facts) - set(goal_facts or [])
        for gf in sorted(extra_sat):
            lines.append(f"  - {gf}: SAT (extra)")
        for gf in sorted(extra_unsat):
            lines.append(f"  - {gf}: UNSAT (extra)")
    elif goal_status_info:
        sat = goal_status_info.get("sat_facts", [])
        unsat = goal_status_info.get("unsat_facts", [])
        all_facts = set(sat + unsat)
        for gf in sorted(all_facts):
            if gf in sat:
                lines.append(f"  - {gf}: SAT")
            else:
                lines.append(f"  - {gf}: UNSAT")
    else:
        lines.append("  (no goal evaluation available)")

    # -- Goal status summary -------------------------------------------------
    lines.append("")
    lines.append("### Goal Status Summary")

    if round_round_facts:
        round_ok = not round_unsat
        lines.append(
            f"Round: {'SAT' if round_ok else 'UNSAT'} — "
            f"{len(round_sat)}/{len(round_round_facts)} round goal(s) met."
        )

    if diagnosis is not None:
        if diagnosis.is_sat:
            lines.append("ALL_SAT — all goals met.")
        elif not diagnosis.unsat_facts:
            lines.append("ALL_SAT — all goals met (auto-inferred from step success).")
        elif len(diagnosis.sat_facts) > 0:
            lines.append(
                f"PARTIAL_UNSAT — {len(diagnosis.sat_facts)} goal(s) met, "
                f"{len(diagnosis.unsat_facts)} goal(s) unmet."
            )
        else:
            lines.append("ALL_UNSAT — no goals met.")
    elif failed_count == 0 and ok_count > 0:
        lines.append("ALL_STEPS_OK — all steps passed. Goals may be SAT.")
    elif failed_count > 0:
        lines.append(f"STEPS_FAILED — {failed_count} step(s) failed.")
    else:
        lines.append("UNKNOWN — no steps executed.")

    return "\n".join(lines)


# ============================================================================
# make_meta_planner_tools
# ============================================================================

def make_meta_planner_tools(
    planner: PlannerDomain,
    step_types: StepTypeCatalog,
    memory_bridge: IMemoryBridge,
    *,
    round_controller: Any = None,
    handoff: Any = None,
    step_executor: Any = None,
    pipeline_config: PipelineConfig | None = None,
    goal_facts: list[str] | None = None,
    round_goal_facts: list[str] | None = None,
    problem: str = "",
    bound_goal: dict[str, str] | None = None,
    initial_planning: bool = False,
    gate: Any = None,
) -> list[StructuredTool]:
    """Build the MetaPlanner tool set.

    Args:
        planner: The PlannerDomain instance whose methods/operators are
            mutated by ``define_method``.
        step_types: The StepType catalog the planner decomposes into.
        memory_bridge: IMemoryBridge for recall/commit.
        round_controller: Optional ``RoundController`` for step building
            and pipeline execution.  When provided, ``create_step`` and
            ``run_custom_pipeline`` delegate to it.
        handoff: Optional ``ICrossRoundHandoff`` for cross-round artifact
            resolution.  When provided, ``get_artifact`` uses it.
        step_executor: Optional StepExecutor (e.g. StepAdapter) for
            ``run_pipeline``.  If None, ``run_pipeline`` will return an
            error explaining it is unavailable.
        pipeline_config: Optional PipelineConfig for ``run_pipeline``.
        goal_facts: Final goal facts to evaluate after ``run_pipeline``
            (the original SUCCESS CRITERIA).
        round_goal_facts: Round-scoped acceptance (e.g. chunk file paths)
            checked via ``FileArtifactChecker`` — reported separately from
            the final facts so the MetaPlanner sees "round done / final done".
        problem: The original problem statement (session root context).
            It is NOT injected as the pipeline's execution input — the
            round goal is bound via ``build_pipeline(problem=...)``.
        initial_planning: If True, execution tools (``run_pipeline``,
            ``get_artifact``) are excluded.  Use this for the pre-loop
            MetaPlanner session that only designs HTN methods.
        gate: Optional per-round ``PlanGate``.  When provided, the tool set
            gains ``submit_plan`` / ``stop`` — the structured plan/stop gate
            ("stop is free; to continue, submit a plan").  ``submit_plan``
            writes the gate's plan text; ``stop`` sets the gate's stop flag
            and optional conclusion.

    Returns:
        List of StructuredTool instances ready for LLM injection.
    """
    # Mutable tracking lists so ``list_methods`` / ``list_operators`` see
    # the current state including methods/operators registered during the
    # MetaPlanner session.
    from quro.algorithm.signature import ProblemSignature

    _registered_methods: list[dict[str, Any]] = []
    _registered_operators: list[dict[str, Any]] = []
    _captured_problem: str = problem
    # Round goal bound by build_pipeline(problem=...) — the ONLY execution
    # input for run_pipeline.  None until build_pipeline is called.
    # ``_captured_problem`` (the original problem) becomes session root
    # context only, never injected as the round's pipeline input.
    _bound_problem: str | None = None
    _goal_facts: list[str] = list(goal_facts or [])
    _round_goal_facts: list[str] = list(round_goal_facts or [])
    # Problem-scoped signature shared by record_decision + recall_memory so
    # decisions are stored and retrieved under the same identity.  This is
    # deliberately separate from the cross-project procedural-learning path
    # used by commit_learning (which uses a generic signature).
    _decision_signature = ProblemSignature.compute(
        _captured_problem or "meta_planner_decision", _goal_facts,
    )

    # ------------------------------------------------------------------
    # define_method
    # ------------------------------------------------------------------
    def define_method(
        task_name: str,
        subtasks_json: str,
        description: str = "",
        method_id: str = "",
        tags: str = "",
    ) -> str:
        """Register a new HTN decomposition method.

        Optional — for HTN verification only.  Prefer ``create_step`` for
        direct step creation (bypasses the HTN solver).

        A method tells the solver how to break down a compound task
        (usually ``"solve_problem"``) into subtasks.  Each subtask must
        reference a registered operator (step type) name.

        HTN schema (see system prompt for full reference):
        - subtasks: JSON array of ``{name, kind, parameters, order_index}``
        - kind: ``"primitive"`` for a step-type step, ``"compound"`` for
          further decomposition.
        - ordering and cost are auto-generated (sequential chain).

        Args:
            task_name: Compound task to decompose (usually ``"solve_problem"``).
            subtasks_json: JSON array of subtask objects, e.g.
                ``[{"name":"implement","kind":"primitive","parameters":{},"order_index":0}]``
            description: Human-readable description of this method.
            method_id: Optional custom method id (auto-generated if empty).
            tags: Optional comma-separated tags.

        Returns:
            Confirmation or error.
        """
        try:
            subtasks = _parse_subtasks_json(subtasks_json)
        except ValueError as e:
            return f"Error parsing subtasks_json: {e}"

        # Validate subtask names against registered operators / step types.
        known_operators = {op["name"] for op in _registered_operators}
        for st in subtasks:
            if st["name"] not in known_operators and not step_types.has(st["name"]):
                return (
                    f"Error: subtask '{st['name']}' is not a registered operator "
                    f"or step type. Available operators: {sorted(known_operators)}. "
                    f"Available step types: {step_types.names()}."
                )

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        method = _build_method_dict(
            task_name=task_name,
            subtasks=subtasks,
            description=description,
            method_id=method_id,
            tags=tag_list,
        )

        ok = planner.register_method(method)
        if not ok:
            return "Error: planner rejected the method (session may not be initialized)."

        _registered_methods.append(method)

        subtask_names = [st["name"] for st in subtasks]
        return (
            f"Method '{method['id']}' registered: {task_name} → "
            f"{' → '.join(subtask_names)}. Cost: {method['cost']}. "
            f"Use build_pipeline to check if solver can now produce a plan."
        )

    # ------------------------------------------------------------------
    # _infer_selected_method (helper for build_pipeline)
    # ------------------------------------------------------------------
    def _infer_selected_method(task_names: list[str]) -> str:
        """Infer which registered method was selected by comparing plan
        task names against each method's subtask names.

        Returns the method ``id`` of the best match, or empty string.
        """
        # Collect methods from local tracking, planner domain, and session state.
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for m in _registered_methods:
            mid = m.get("id", "")
            if mid not in seen:
                seen.add(mid)
                candidates.append(m)
        for m in planner.get_custom_methods():
            mid = m.get("id", "")
            if mid not in seen:
                seen.add(mid)
                candidates.append(m)
        if planner.session is not None:
            state = planner.session.state or {}
            for m in state.get("methods", []):
                if isinstance(m, dict) and m.get("id", "") not in seen:
                    seen.add(m.get("id", ""))
                    candidates.append(m)

        best_id = ""
        best_score = 0
        for m in candidates:
            subtask_names = [
                st.get("name", "") for st in m.get("subtasks", [])
            ]
            # Score: count of matching names (order-independent).
            score = sum(1 for tn in task_names if tn in subtask_names)
            if score > best_score:
                best_score = score
                best_id = m.get("id", "")

        return best_id

    # ------------------------------------------------------------------
    # build_pipeline
    # ------------------------------------------------------------------
    def build_pipeline(problem: str) -> str:
        """Run the HTN solver and build a pipeline from its plan.

        Internal use — manual invocation is only for HTN verification.
        The framework's ``g`` handles this internally.  Prefer
        ``create_step`` + ``run_custom_pipeline`` for direct step creation.

        Binds ``problem`` as this round's job goal — it becomes the ONLY
        execution input for the subsequent ``run_pipeline()`` call.

        Calls ``planner.solve()`` then ``planner.pef_to_pipeline()``.
        Returns a summary of the resulting pipeline (step count, order,
        and which method was selected) or reports an empty plan with
        diagnostic information about registered methods.

        Use this after calling ``define_method`` to
        verify that the solver can now produce a plan.

        Args:
            problem: The round job goal (REQUIRED).  Embed CONTEXT
                (workspace / codebase root / chunk dirs) into it before
                calling.

        Returns:
            Pipeline summary (including the bound problem) or error.
        """
        nonlocal _bound_problem

        _bound_problem = problem
        if bound_goal is not None:
            bound_goal["problem"] = problem
        # Always re-init: each round gets a fresh solver session (ADR-003
        # FR-4 — one session per round).  Custom methods (registered via
        # define_method) filter out the default sequential method so they
        # are not shadowed.
        custom = [
            m for m in _registered_methods
            if m.get("task_name") == "solve_problem"
            and m.get("id") != "solve_problem_sequential"
        ]
        logger.info(
            "build_pipeline: re-initialising planner with %d custom method(s): %s",
            len(custom),
            [(m.get("id"), [st.get("name") for st in m.get("subtasks", [])])
             for m in custom],
        )
        planner.init(
            problem,
            step_types,
            goal_facts=[],
            method_definitions=custom,
        )

        pef = planner.solve()
        # Diagnostic: compare PEF task count against expected subtask count.
        if pef:
            tasks = pef.get("executionGraph", {}).get("tasks", [])
            task_names = [t.get("name", "?") for t in tasks]
            if custom:
                expected = [st.get("name") for m in custom
                            for st in m.get("subtasks", [])]
                if len(tasks) != len(expected):
                    logger.warning(
                        "build_pipeline: PEF task count mismatch — "
                        "expected %d tasks (%s), got %d tasks (%s)",
                        len(expected), expected, len(tasks), task_names,
                    )
        if not pef:
            # Build diagnostic: show registered methods so MetaPlanner
            # can understand why no method matched.
            diag_lines = [
                "PLAN-UNSAT: The HTN solver still cannot produce a plan.",
            ]

            # Collect method info from local tracking + session state.
            method_entries: list[dict[str, Any]] = []
            seen: set[str] = set()
            for m in _registered_methods:
                mid = m.get("id", "?")
                if mid not in seen:
                    seen.add(mid)
                    method_entries.append(m)
            for m in planner.get_custom_methods():
                mid = m.get("id", "?")
                if mid not in seen:
                    seen.add(mid)
                    method_entries.append(m)
            if planner.session is not None:
                state = planner.session.state or {}
                for m in state.get("methods", []):
                    if isinstance(m, dict) and m.get("id", "") not in seen:
                        seen.add(m.get("id", ""))
                        method_entries.append(m)

            if method_entries:
                diag_lines.append(f"Registered methods ({len(method_entries)}):")
                for m in method_entries:
                    subtask_names = [
                        st.get("name", "?") for st in m.get("subtasks", [])
                    ]
                    preconds = m.get("preconditions", [])
                    precond_str = (
                        f"preconditions={preconds}" if preconds
                        else "preconditions=[] (always matches)"
                    )
                    source = (
                        "[DEFAULT]" if m.get("id") == "solve_problem_sequential"
                        else "[custom]"
                    )
                    diag_lines.append(
                        f"  - {m.get('id', '?')} {source}: "
                        f"{' → '.join(subtask_names)} "
                        f"(cost={m.get('cost', '?')}, {precond_str})"
                    )
            else:
                diag_lines.append(
                    "No methods registered.  Use define_method to add "
                    "decomposition paths."
                )

            diag_lines.append(
                "Check available operators with list_operators. "
                "Try a different decomposition."
            )
            return "\n".join(diag_lines)

        tasks = pef.get("executionGraph", {}).get("tasks", [])
        task_names = [t.get("name", "?") for t in tasks]

        # Try to identify which registered method was selected by
        # matching subtask names against the plan tasks.
        selected_id = _infer_selected_method(task_names)
        method_note = ""
        if selected_id:
            method_note = f"\nSelected method: {selected_id}"
        else:
            method_note = "\nSelected method: unknown (check list_methods)"

        return (
            f"Bound problem: {problem}\n"
            f"Plan produced: {len(tasks)} step(s).\n"
            f"Order: {' → '.join(task_names)}"
            f"{method_note}\n"
            f"PEF ready. Use run_pipeline to execute."
        )

    # ------------------------------------------------------------------
    # run_pipeline
    # ------------------------------------------------------------------
    def run_pipeline() -> str:
        """Solve, build, and execute the pipeline, returning a summary.

        Advisory — the framework's round outcome is authoritative.  Use this
        only to prototype a candidate decomposition before committing it as
        your plan.

        Executes the round goal bound by the previous ``build_pipeline``
        call — there is no ``problem`` parameter here; the pipeline can
        only run what was built.

        This is a **summary-only** tool — it returns a structured summary
        (status, per-step one-line results, goal evaluation), NOT raw
        step artifacts.  This prevents context explosion in the MetaPlanner.

        Recursion guard: steps whose step type is ``meta-planner`` are rejected.

        Returns:
            Structured pipeline summary, or an error if no pipeline has
            been built yet.
        """
        if step_executor is None:
            return "Error: run_pipeline unavailable — no step_executor configured."

        if _bound_problem is None:
            return (
                "Error: no pipeline built — call "
                "build_pipeline(problem=...) first."
            )

        # Re-init to filter out default method if custom methods exist
        # (same rationale as build_pipeline).
        custom = [
            m for m in _registered_methods
            if m.get("task_name") == "solve_problem"
            and m.get("id") != "solve_problem_sequential"
        ]
        if custom:
            logger.info(
                "run_pipeline: re-initialising with %d custom method(s): %s",
                len(custom),
                [(m.get("id"), [st.get("name") for st in m.get("subtasks", [])])
                 for m in custom],
            )
            planner.init(
                _bound_problem,
                step_types,
                goal_facts=[],
                method_definitions=custom,
            )

        pef = planner.solve()
        if pef:
            tasks = pef.get("executionGraph", {}).get("tasks", [])
            task_names = [t.get("name", "?") for t in tasks]
            if custom:
                expected = [st.get("name") for m in custom
                            for st in m.get("subtasks", [])]
                if len(tasks) != len(expected):
                    logger.warning(
                        "run_pipeline: PEF task count mismatch — "
                        "expected %d tasks (%s), got %d tasks (%s)",
                        len(expected), expected, len(tasks), task_names,
                    )
        if not pef:
            return "PLAN-UNSAT: solver returned empty plan.  Try define_method first."

        # Recursion guard: reject any step whose step type is meta-planner.
        tasks = pef.get("executionGraph", {}).get("tasks", [])
        for task in tasks:
            if task.get("name") == "meta-planner":
                return (
                    "Error: pipeline contains a 'meta-planner' step. "
                    "MetaPlanner cannot execute itself (recursion guard). "
                    "Remove this step type and try again."
                )

        pipeline = planner.pef_to_pipeline(
            pef, default_config=pipeline_config,
        )


        runner = PipelineRunner(
            pipeline,
            step_executor=step_executor,
            config=pipeline_config or PipelineConfig(),
            step_types=step_types,
        )

        result = runner.run(_bound_problem)

        # Carry produced artifacts into the cross-round handoff so a
        # subsequent get_artifact resolves them (same resolution layer as the
        # loop's own f() execution).
        if handoff is not None:
            from quro.core.cross_round_handoff import carry_result_artifacts

            round_idx = (
                getattr(round_controller, "round_idx", 0)
                if round_controller is not None else 0
            )
            carry_result_artifacts(handoff, round_idx, result)

        return _summarize_pipeline_result(
            result,
            goal_facts=goal_facts,
            round_goal_facts=_round_goal_facts,
        )

    # ------------------------------------------------------------------
    # recall_memory
    # ------------------------------------------------------------------
    def recall_memory(query: str, limit: int = 3) -> str:
        """Search memory for prior design decisions and learned patterns.

        Returns two categories:
          1. problem-scoped design decisions (recorded via
             ``record_decision``), and
          2. cross-project patterns (``pattern:goal_unmet``,
             ``pattern:htn_unsolvable``).

        Args:
            query: Search query string (reserved; the problem-scoped
                decision store is keyed by the problem signature).
            limit: Maximum results per category (default 3).

        Returns:
            JSON-formatted list of memory entries, or empty list.
        """
        # (a) Problem-scoped design decisions.
        decisions = memory_bridge.recall(
            _decision_signature, memory_types=["decision"], limit=limit,
        )

        # (b) Cross-project pattern recall.
        patterns = memory_bridge.recall_patterns(
            pattern_tags=[
                "pattern:goal_unmet",
                "pattern:htn_unsolvable",
                "pattern:resource_exhaustion",
            ],
            limit=limit,
        )

        simplified: list[dict[str, Any]] = []
        for d in decisions:
            simplified.append({
                "kind": "decision",
                "type": d.get("type", "decision"),
                "tags": d.get("tags", []),
                "body": str(d.get("body", ""))[:400],
            })
        for p in patterns:
            simplified.append({
                "kind": "pattern",
                "type": p.get("type", "unknown"),
                "tags": p.get("tags", []),
                "body": str(p.get("body", ""))[:300],
            })

        if not simplified:
            return (
                "[]  (no prior decisions or patterns found — expected on "
                "first run)"
            )

        return json.dumps(simplified, indent=2)

    # ------------------------------------------------------------------
    # record_decision
    # ------------------------------------------------------------------
    def record_decision(body: str, tags: str = "") -> str:
        """Record your design decision / rationale for the current plan.

        This is NOT "learning" — it stores the raw background of *why* you
        designed these methods this round, scoped to the current
        problem.  A later escalation can selectively retrieve it via the
        dedicated ``decision`` HINT and ``recall_memory``.

        Args:
            body: The decision / rationale — which step types, why this
                ordering, what alternatives you rejected.
            tags: Comma-separated tags (e.g. ``"method-split,ordering"``).

        Returns:
            Memory entry id or error.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        if "decision" not in tag_list:
            tag_list.insert(0, "decision")

        entry_id = memory_bridge.distill(
            signature=_decision_signature,
            round_index=0,
            body=body,
            tags=tag_list,
            memory_type="decision",
            importance=0.8,
        )
        memory_bridge.flush()

        if entry_id:
            return f"Decision recorded: {entry_id}"
        return "Decision recorded (no id returned — backend may be unavailable)."

    # ------------------------------------------------------------------
    # commit_learning
    # ------------------------------------------------------------------
    def commit_learning(body: str, tags: str = "") -> str:
        """Persist a learned pattern for future sessions.

        Writes to quro-memory as type ``"procedural"`` with high
        importance (0.8).

        Args:
            body: The learning content — describe what method you
                created and why, so future MetaPlanners can reuse it.
            tags: Comma-separated tags (e.g.
                ``"pattern:htn_unsolvable,l1_repair"``).

        Returns:
            Memory entry id or error.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        if not tag_list:
            tag_list = ["l1_repair"]

        # Add standard pattern tags.
        for pattern_tag in ("pattern:htn_unsolvable", "pattern:goal_unmet"):
            if pattern_tag not in tag_list:
                tag_list.append(pattern_tag)

        # Procedural learning is cross-project: persist under a generic
        # signature so recall_patterns() finds it across problems.  (Problem-
        # scoped decisions use record_decision instead.)
        sig = ProblemSignature.compute("meta_planner_learning", [])

        entry_id = memory_bridge.distill(
            signature=sig,
            round_index=0,
            body=body,
            tags=tag_list,
            memory_type="procedural",
            importance=0.8,
        )
        memory_bridge.flush()

        if entry_id:
            return f"Learning committed: {entry_id}"
        return "Learning committed (no id returned — backend may be unavailable)."

    # ------------------------------------------------------------------
    # list_methods
    # ------------------------------------------------------------------
    def list_methods() -> str:
        """List all HTN methods currently registered in the planner.

        Includes the default sequential method (tagged ``[DEFAULT]``)
        plus any methods registered via ``define_method`` during this
        MetaPlanner session.  Each entry includes ``source``
        (``meta-planner``, ``system``, or ``system [DEFAULT]``) and
        ``preconditions`` so the MetaPlanner can understand method
        priority and matching behaviour.

        Returns:
            JSON list of method summaries.
        """
        seen_ids: set[str] = set()
        all_methods: list[dict[str, Any]] = []

        # MetaPlanner-registered methods first (source = meta-planner).
        for m in _registered_methods:
            mid = m.get("id", "")
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_methods.append({**m, "_source": "meta-planner"})

        # PlannerDomain-registered methods (get_custom_methods).
        for m in planner.get_custom_methods():
            mid = m.get("id", "")
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_methods.append({**m, "_source": "meta-planner"})

        # Session state methods (including the default sequential method).
        if planner.session is not None:
            state = planner.session.state or {}
            state_methods = state.get("methods", [])
            for m in state_methods:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id", "")
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    if mid == "solve_problem_sequential":
                        source = "system [DEFAULT]"
                    else:
                        source = "system"
                    all_methods.append({**m, "_source": source})

        if not all_methods:
            return (
                "[]  (no methods visible — the planner may use its default "
                "sequential decomposition internally. "
                "Use define_method to add custom decompositions.)"
            )

        simplified = []
        for m in all_methods:
            subtask_names = [st.get("name", "?") for st in m.get("subtasks", [])]
            simplified.append({
                "id": m.get("id", "?"),
                "task_name": m.get("task_name", "?"),
                "subtasks": subtask_names,
                "preconditions": m.get("preconditions", []),
                "cost": m.get("cost", "?"),
                "description": m.get("description", ""),
                "source": m.get("_source", "unknown"),
            })

        return json.dumps(simplified, indent=2)

    # ------------------------------------------------------------------
    # list_operators
    # ------------------------------------------------------------------
    def list_operators() -> str:
        """List all HTN primitive operators currently registered.

        Operators correspond to step types in the catalog.  Each operator
        has an effect ``{name}_done`` that the solver uses for
        dependency tracking.

        Returns:
            JSON list of operator summaries (including skill_pool).
        """
        from quro.core.catalog_adapter import StepCatalogAdapter

        catalog_view = StepCatalogAdapter(step_types)
        summaries = catalog_view.describe_operators()

        # Also include operators registered in the planner session.
        session_ops: list[dict[str, Any]] = []
        if planner.session is not None:
            state = planner.session.state or {}
            state_operators = state.get("operators", [])
            for op in state_operators:
                if isinstance(op, dict):
                    session_ops.append(op)

        if not summaries and not session_ops:
            return "[]  (no operators registered)"

        simplified = []
        for s in summaries:
            simplified.append({
                "name": s.name,
                "description": s.description,
                "cost": s.cost,
                "skill_pool": list(s.skill_pool),
                "access": s.access,
            })
        for op in session_ops:
            simplified.append({
                "name": op.get("name", "?"),
                "description": op.get("description", ""),
                "cost": op.get("cost", "?"),
                "skill_pool": list(op.get("skill_pool", [])),
                "access": op.get("access", ""),
            })

        return json.dumps(simplified, indent=2)

    # ------------------------------------------------------------------
    # get_artifact
    # ------------------------------------------------------------------
    def get_artifact(artifact_id: str) -> str:
        """Retrieve the full content of an artifact produced by a pipeline step.

        Looks up artifacts from the cross-round handoff.  Returns the
        artifact's full record including summary, kind, body, and the
        step that produced it.

        Args:
            artifact_id: The artifact ID to retrieve (e.g. ``"art_a1b2c3d4"``).

        Returns:
            JSON-formatted artifact record or error message.
        """
        if handoff is None:
            return (
                "Error: no cross-round handoff configured. "
                "Cannot retrieve artifacts."
            )

        art = handoff.resolve_artifact(artifact_id)
        if art is None:
            return f"Error: artifact not found: {artifact_id}"

        record = {
            "artifact_id": artifact_id,
            "kind": getattr(art, "kind", "") if hasattr(art, "kind") else "",
            "summary": getattr(art, "summary", "") if hasattr(art, "summary") else "",
            "body": str(getattr(art, "body", art))[:2000],
        }
        return json.dumps(record, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # submit_plan / stop — the structured plan/stop gate
    # ------------------------------------------------------------------
    def submit_plan(plan_text: str) -> str:
        """Submit (or overwrite) this round's plan — REQUIRED to continue.

        This is the hard gate: the loop will not execute until a plan is
        submitted.  A verbatim resubmission is valid — it is the explicit
        statement "keep the current plan".
        """
        if gate is None:
            return "Error: plan gate not configured — cannot submit a plan."
        gate.plan_text = plan_text
        return (
            f"Plan submitted ({len(plan_text)} chars). "
            f"The runtime will execute it this round."
        )

    def stop(conclusion: str = "") -> str:
        """Stop now (free stop) — no further plan is required.

        Optional ``conclusion`` text is recorded as the user-facing
        conclusion.
        """
        if gate is None:
            return "Error: plan gate not configured — cannot stop."
        gate.stop = True
        gate.conclusion = conclusion
        return "Stopped. The runtime will return the report."

    # ------------------------------------------------------------------
    # create_step — direct step creation (bypasses HTN solver)
    # ------------------------------------------------------------------
    def create_step(
        step_id: str,
        objective: str,
        step_type: str = "",
        depends_on: str = "",
        skills: str = "",
        expected_output: str = "",
        validation: str = "",
    ) -> str:
        """Create a step specification directly (bypasses HTN solver).

        This is the **primary** tool for building pipelines.  Each step
        gets its own ``step_id``, ``objective``, and ``depends_on`` chain —
        the HTN solver is not involved.

        Args:
            step_id: Unique step identifier (e.g. "survey_modules").
            objective: The step's objective — what it should accomplish.
            step_type: Step type name from the catalog (e.g. "survey_module").
            depends_on: Comma-separated step_ids this step depends on.
            skills: Comma-separated skill names from the step type's pool.
            expected_output: Description of expected output.
            validation: Criteria for step completion.

        Returns:
            Confirmation or error.
        """
        if round_controller is None:
            return "Error: no RoundController configured — cannot create steps."
        result = round_controller.build_step(
            step_id=step_id,
            objective=objective,
            step_type=step_type,
            depends_on=[d.strip() for d in depends_on.split(",") if d.strip()] if depends_on else [],
            skills=[s.strip() for s in skills.split(",") if s.strip()] if skills else [],
            expected_output=expected_output,
            validation=validation,
        )
        if not result.dependency_check.ok:
            return f"Error: {'; '.join(result.dependency_check.errors)}"
        return (
            f"Step '{step_id}' created (type={step_type or 'none'}, "
            f"features={result.resolved_features}, "
            f"warnings={result.warnings or 'none'})."
        )

    # ------------------------------------------------------------------
    # setup_step — update a manual step's objective / deps / validation
    # ------------------------------------------------------------------
    def setup_step(
        step_id: str,
        objective: str = "",
        depends_on: str = "",
        expected_output: str = "",
        validation: str = "",
    ) -> str:
        """Update an existing manually-created step's fields.

        Use this to refine step objective or adjust dependencies as new
        information emerges across rounds.

        Args:
            step_id: The step to update.
            objective: New objective text (empty = keep current).
            depends_on: New comma-separated dependency list (empty = keep).
            expected_output: New expected output description.
            validation: New validation criteria.

        Returns:
            Confirmation or error.
        """
        if round_controller is None:
            return "Error: no RoundController configured — cannot update steps."
        target = round_controller.builder._pending.get(step_id)
        if target is None:
            available = list(round_controller.builder._pending.keys())
            return (
                f"Error: step '{step_id}' not found in manual steps. "
                f"Available: {available}"
            )

        fields: dict[str, Any] = {}
        if objective:
            fields["objective"] = objective
        if depends_on is not None:
            fields["depends_on"] = (
                [d.strip() for d in depends_on.split(",") if d.strip()]
                if depends_on else []
            )
        if expected_output:
            fields["expected_output"] = expected_output
        if validation:
            fields["validation"] = validation

        round_controller.builder.update(step_id, **fields)

        changes = []
        if objective:
            changes.append("objective")
        if depends_on is not None:
            changes.append("depends_on")
        if expected_output:
            changes.append("expected_output")
        if validation:
            changes.append("validation")
        return f"Step '{step_id}' updated: {', '.join(changes)}"

    # ------------------------------------------------------------------
    # run_custom_pipeline — execute manually-created steps
    # ------------------------------------------------------------------
    def run_custom_pipeline() -> str:
        """Execute the pipeline built from manually-created steps.

        Assembles a Pipeline from steps created via ``create_step`` and
        runs it.  Returns a structured summary (same format as
        ``run_pipeline``).

        This bypasses the HTN solver entirely — steps are executed in
        dependency order as created by the MetaPlanner.

        Returns:
            Pipeline execution summary, or error.
        """
        if round_controller is None:
            return "Error: run_custom_pipeline unavailable — no RoundController."
        if not round_controller.scope.pending_steps:
            return "Error: no manual steps created. Use create_step first."

        result = round_controller.execute()
        return _summarize_pipeline_result(result)

    # ------------------------------------------------------------------
    # list_manual_steps — list manually created steps
    # ------------------------------------------------------------------
    def list_manual_steps() -> str:
        """List all steps created via ``create_step``.

        Returns a JSON summary of each manual step including its
        dependencies and validation criteria.
        """
        if round_controller is None:
            return "[]  (no RoundController configured)"
        pending = round_controller.scope.pending_steps
        if not pending:
            return "[]  (no manual steps — use create_step to add)"
        items = []
        for step_id, result in pending.items():
            spec = result.spec
            items.append({
                "step_id": step_id,
                "objective": spec.objective[:120] if spec else "",
                "step_type": spec.step_type if spec else "",
                "depends_on": list(spec.depends_on) if spec else [],
                "skills": list(spec.skills) if spec else [],
            })
        return json.dumps(items, indent=2)

    # ------------------------------------------------------------------
    # Build StructuredTool list.
    # ------------------------------------------------------------------
    # Gate tools — the structured plan/stop gate, always available when wired.
    _gate_tools: list[StructuredTool] = []
    if gate is not None:
        _gate_tools = [
            StructuredTool.from_function(func=submit_plan),
            StructuredTool.from_function(func=stop),
        ]

    # Design tools — always available.
    _design_tools: list[StructuredTool] = [
        StructuredTool.from_function(func=define_method),
        StructuredTool.from_function(func=build_pipeline),
        StructuredTool.from_function(func=list_methods),
        StructuredTool.from_function(func=list_operators),
    ]

    # Step creation tools — direct step creation bypassing HTN solver.
    _step_creation_tools: list[StructuredTool] = [
        StructuredTool.from_function(func=create_step),
        StructuredTool.from_function(func=setup_step),
        StructuredTool.from_function(func=list_manual_steps),
    ]

    if initial_planning:
        # Initial planning: design tools + recall/record so the MetaPlanner
        # can review and persist its design rationale (RC-4).  No execution
        # or termination tools — the MetaPlanner is designing methods, not
        # running or terminating the pipeline.
        _memory_tools: list[StructuredTool] = [
            StructuredTool.from_function(func=recall_memory),
            StructuredTool.from_function(func=record_decision),
        ]
        return _gate_tools + _design_tools + _step_creation_tools + _memory_tools

    # Full tool set for L1 escalation.
    _execution_tools: list[StructuredTool] = [
        StructuredTool.from_function(func=run_pipeline),
        StructuredTool.from_function(func=run_custom_pipeline),
        StructuredTool.from_function(func=get_artifact),
        StructuredTool.from_function(func=recall_memory),
        StructuredTool.from_function(func=record_decision),
        StructuredTool.from_function(func=commit_learning),
    ]

    return _gate_tools + _design_tools + _step_creation_tools + _execution_tools
