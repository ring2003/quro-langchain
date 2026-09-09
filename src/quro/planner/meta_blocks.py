"""MetaPlannerBlockProvider — manages block content for MetaPlanner sessions.

Implements ``IPromptBlockProvider``.  Each block is a named content
fragment that feeds into the Context Layer's ``ISessionContext.assemble``
pipeline.  Blocks are dynamic — they change per escalation round — but
their structure is stable, so the Context Layer can trim and account
tokens per-block.

Block catalogue
---------------

===================== ======= ===========================================
Block                  Kind    Description
===================== ======= ===========================================
role                  system  Core identity description (who MetaPlanner is)
constraints           system  Rules, constraints, and few-shot examples
tools                 system  Available tools with one-line descriptions
hints                 system  Dynamic HINT lines (see MetaHintsBuilder)
context               user    Escalation mode, problem, round, attempts
diagnosis             user    UNSAT diagnosis (level, facts, cause)
pipeline              user    Pipeline execution report (self-contained)
===================== ======= ===========================================

Design principle: blocks are **managed** here but **assembled** by the
Context Layer.  This module owns block content; ``ISessionContext`` owns
block rendering (message → dict, trim, token ledger).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quro.context.protocols import IPromptBlockProvider


# ============================================================================
# Block content constants
# ============================================================================

META_PLANNER_ROLE_BLOCK = """You are a MetaPlanner — an LLM that repairs HTN (Hierarchical Task Network) plans
when the deterministic solver fails to make progress.

## Your Role

You operate in one of two escalation modes (injected in the user context block):

- **define_method** — the HTN solver returned empty (PLAN-UNSAT).  You must design
  a new decomposition method so the solver can produce a plan.
- **analyze_and_fix** — the pipeline executed but goals remain unmet after
  multiple rounds (VERIFY-UNSAT stall).  You must diagnose what went wrong
  and propose a targeted fix.

## HTN Schema (reference)

- **PrimitiveOperator**: {name, parameters, cost} — maps to a step-type step.
  Effect is ``{name}_done``.
- **Method**: {task_name, subtasks, ordering, preconditions} — decomposes
  a task into subtasks.  ``task_name`` is usually ``"solve_problem"``.
- **ordering**: list of [predecessor_index, successor_index] pairs.

When you call ``define_method``, the system registers your method AND
automatically filters out the default sequential method so yours takes
precedence.  You do NOT need a separate ``replace_default_method`` call.
"""

META_PLANNER_CONSTRAINTS_BLOCK = """## Rules

1. Only use the tools provided.  You have NO filesystem access.
2. Do NOT create a step type named ``meta-planner`` — it is forbidden.
3. Always include a **step-specification step type** as the **first subtask**
   in your method — ``plan``.  It creates the step
   specification other step types depend on.
4. After calling ``define_method``, call ``build_pipeline`` to verify the
   solver produces a plan.
5. If ``build_pipeline`` returns empty, try a different approach.
6. Use ``record_decision`` to record your design rationale (why these
   step types / this ordering).  Only call ``commit_learning`` when you have
   a proven, reusable cross-project pattern.

## Reading the Pipeline Execution Report

After ``run_pipeline`` returns, you will see a structured report:

```
### Step Summary (N/N OK)
  - step_id: OK/FAILED — artifact description
### Goal Status
  - goal_name: SAT/UNSAT — evidence
### Verdict: ALL_SAT | PARTIAL_UNSAT | ALL_UNSAT
```

**Decision tree:**
- all goals SAT → record the outcome and stop
- some goals UNSAT → read which goal is UNSAT → define_method → re-run
- no goals met → fundamental problem → major redesign

## Common Anti-patterns

| Wrong | Why | Right |
|-------|-----|-------|
| "Let implement run tests" | Violates closure context | Decompose a separate implement step |
| Skip plan in method | No step specification for executors | Always include plan as subtask 0 |
| Define method without build_pipeline | Don't know if solver can use it | Always build_pipeline after define_method |
"""

META_PLANNER_TOOLS_BLOCK = """## TOOLS

- submit_plan(plan_text) — submit/overwrite the current plan (REQUIRED to continue)
- stop(conclusion) — stop now (free stop), optional user-facing conclusion
- list_operators — list registered step types/operators
- list_methods — list registered HTN methods
- define_method(task_name, subtasks_json, description) — define a new HTN decomposition
- build_pipeline — solve + verify plan without executing
- run_pipeline — solve + build + execute pipeline, returns self-contained report
- create_step(step_id, objective, step_type, depends_on, skills, expected_output, validation) — create a step directly (bypasses HTN solver)
- setup_step(step_id, objective, depends_on, expected_output, validation) — update an existing manual step
- list_manual_steps — list steps created via create_step
- run_custom_pipeline — execute manually-created steps (bypasses HTN solver)
- get_artifact(artifact_id) — retrieve full artifact content by ID
- recall_memory(query, limit) — search prior decisions + learned patterns
- record_decision(body, tags) — record your design rationale (NOT learning)
- commit_learning — persist a proven cross-project pattern for future sessions
"""


# ============================================================================
# MetaPlannerBlockProvider
# ============================================================================


@dataclass
class MetaPlannerBlockProvider(IPromptBlockProvider):
    """Block provider for MetaPlanner sessions.

    Produces named blocks that the Context Layer assembles into the final
    message list.  Dynamic blocks (context, diagnosis, pipeline) are set
    via properties before each invocation; static blocks (identity, constraints,
    tools) are constant.

    Usage::

        provider = MetaPlannerBlockProvider()
        provider.set_context(
            escalation_mode="analyze_and_fix",
            problem="Build a calculator...",
            round_idx=2,
            l1_attempts=0,
        )
        provider.set_diagnosis(diagnosis)
        provider.set_pipeline_results(all_results)
        provider.set_hints(hints)

        blocks = provider.get_blocks("meta-planner")
        # blocks["role"], blocks["constraints"], blocks["tools"],
        # blocks["hints"], blocks["context"], blocks["diagnosis"],
        # blocks["pipeline"]
    """

    # -- Dynamic block state -------------------------------------------------

    _context_str: str = ""
    _diagnosis_str: str = ""
    _pipeline_str: str = ""
    _hints_str: str = ""
    _ppf_str: str = ""
    _plan_str: str = ""
    _rounds_str: str = ""

    # -- Static blocks -------------------------------------------------------

    _role_block: str = META_PLANNER_ROLE_BLOCK
    _constraints_block: str = META_PLANNER_CONSTRAINTS_BLOCK
    _tools_block: str = META_PLANNER_TOOLS_BLOCK

    # -- IPromptBlockProvider -------------------------------------------------

    def get_blocks(self, principal: str) -> dict[str, str]:
        """Return all blocks for *principal*.

        Only ``"meta-planner"`` is supported.
        """
        if principal != "meta-planner":
            return {}
        return {
            "role": self._role_block,
            "constraints": self._constraints_block,
            "tools": self._tools_block,
            "hints": self._hints_str,
            "context": self._context_str,
            "diagnosis": self._diagnosis_str,
            "pipeline": self._pipeline_str,
            "ppf": self._ppf_str,
            "plan": self._plan_str,
            "rounds": self._rounds_str,
        }

    def list_principals(self) -> list[str]:
        return ["meta-planner"]

    def list_blocks(self, principal: str) -> list[str]:
        if principal != "meta-planner":
            return []
        return [
            "role", "constraints", "tools", "hints",
            "context", "diagnosis", "pipeline",
            "ppf", "plan", "rounds",
        ]

    # -- Setters for dynamic blocks ------------------------------------------

    def set_context(
        self,
        *,
        escalation_mode: str,
        problem: str,
        round_idx: int,
        l1_attempts: int,
        run_id: str | None = None,
    ) -> None:
        """Set the user.context block."""
        parts: list[str] = [
            f"## Escalation Mode: {escalation_mode}",
            f"**Problem:** {problem[:500]}",
            f"Round: {round_idx + 1}",
            f"L1 attempts so far: {l1_attempts}",
        ]
        if run_id:
            parts.append(f"Run: {run_id}")
        self._context_str = "\n".join(parts)

    def set_diagnosis(self, diagnosis: Any) -> None:
        """Set the user.diagnosis block from an UNSATDiagnosis."""
        parts: list[str] = [
            "## UNSAT Diagnosis",
            f"Level: {diagnosis.level.value}",
            f"Cause: {diagnosis.cause}",
        ]
        if diagnosis.unsat_facts:
            parts.append(f"UNSAT facts: {diagnosis.unsat_facts}")
        if diagnosis.sat_facts:
            parts.append(f"SAT facts: {diagnosis.sat_facts}")
        if diagnosis.step_id:
            parts.append(f"Failing step: {diagnosis.step_id}")
        if diagnosis.step_error:
            parts.append(f"Step error: {diagnosis.step_error[:200]}")
        self._diagnosis_str = "\n".join(parts)

    def set_pipeline_results(
        self,
        all_results: dict[str, Any] | None,
        rounds: list[dict[str, Any]] | None = None,
    ) -> None:
        """Set the user.pipeline block from accumulated step results.

        Produces a compact summary: one line per step with status and
        artifact IDs, plus key state values.  When ``rounds`` is provided
        (a list of per-round step-result dicts), the summary is grouped by
        round so history is traceable (RC-5).
        """
        if rounds:
            parts: list[str] = ["## Pipeline Results (by round)"]
            for r_idx, round_results in enumerate(rounds):
                parts.append(f"### Round {r_idx + 1}")
                if not round_results:
                    parts.append("  (no steps)")
                    continue
                parts.extend(self._render_step_lines(round_results))
            self._pipeline_str = "\n".join(parts)
            return

        if not all_results:
            self._pipeline_str = "## Pipeline Results\n(no prior results)"
            return

        parts: list[str] = ["## Pipeline Results"]
        parts.extend(self._render_step_lines(all_results))
        self._pipeline_str = "\n".join(parts)

    @staticmethod
    def _render_step_lines(results: dict[str, Any]) -> list[str]:
        """Render one-line-per-step summaries for a single round."""
        lines: list[str] = []
        for step_id, sr in results.items():
            status = "OK" if sr.ok else f"FAIL: {getattr(sr, 'error', '') or 'unknown'}"
            art_ids: list[str] = []
            for artifact in (getattr(sr, "artifacts", []) or []):
                if isinstance(artifact, dict):
                    aid = artifact.get("artifact_id", "")
                    if aid:
                        art_ids.append(aid)
            art_str = f"  artifacts: [{', '.join(art_ids)}]" if art_ids else ""
            lines.append(f"  {step_id}: {status}{art_str}")

            # One-line artifact summaries.
            for i, artifact in enumerate(getattr(sr, "artifacts", []) or []):
                if isinstance(artifact, dict):
                    summary = str(artifact.get("summary", ""))[:120]
                    if summary:
                        lines.append(f"    [{i}] {summary}")

            # Key state keys.
            if isinstance(getattr(sr, "state", None), dict):
                for key in ("plan", "interpretation"):
                    val = sr.state.get(key)
                    if val:
                        lines.append(f"    state.{key}: {str(val)[:150]}")
        return lines

    def set_hints(self, hints: str) -> None:
        """Set the system.hints block."""
        self._hints_str = hints

    def set_ppf(self, ppf: Any) -> None:
        """Set the user.ppf block (the problem description, static)."""
        if ppf is None:
            self._ppf_str = ""
            return
        lines = ["## Problem (PPF)", f"Problem: {ppf.problem}" if getattr(ppf, "problem", "") else "Problem: (none)"]
        if getattr(ppf, "goals", None):
            lines.append("Goals: " + ", ".join(ppf.goals))
        if getattr(ppf, "constraints", None):
            lines.append("Constraints: " + ", ".join(ppf.constraints))
        if getattr(ppf, "plan", None):
            lines.append("Exploration skeleton: " + " | ".join(ppf.plan))
        self._ppf_str = "\n".join(lines)

    def set_plan(self, plan_text: str | None) -> None:
        """Set the user.plan block in *echo voice*.

        The plan is the agent's own draft — it is injected as "this is the
        plan you wrote last round, revisable", never as an imperative
        instruction (``meta-plan-prompt-and-ppf.md`` §4.3).
        """
        if not plan_text:
            self._plan_str = ""
            return
        self._plan_str = (
            "## Plan (your draft — revisable)\n"
            "This is the plan you wrote last round.  You may keep it verbatim "
            "(= \"keep the current plan\") or overwrite it.\n\n"
            f"{plan_text}"
        )

    def set_rounds(self, rounds: list[Any]) -> None:
        """Set the user.rounds block (framework-owned outcome trace).

        Historical rounds are compressed to ``sequence — outcome``; the
        current round is expanded (the caller decides which round is current).
        """
        if not rounds:
            self._rounds_str = ""
            return
        lines = ["## Rounds (outcomes)"]
        for r in rounds:
            gs = getattr(r, "goal_status", {}) or {}
            sat = gs.get("sat_facts", [])
            unsat = gs.get("unsat_facts", [])
            lines.append(
                f"- Round {getattr(r, 'round_idx', 0) + 1}: "
                f"sat={sat}, unsat={unsat}"
            )
        self._rounds_str = "\n".join(lines)

    # -- Block visibility for the Coordinator --------------------------------

    @staticmethod
    def visible_blocks() -> set[str]:
        """Return the set of block paths a MetaPlanner session should see."""
        return {
            "role",
            "constraints",
            "tools",
            "hints",
            "context",
            "diagnosis",
            "pipeline",
            "ppf",
            "plan",
            "rounds",
        }
