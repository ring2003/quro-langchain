"""MetaHintsBuilder — generates compact HINT lines for MetaPlanner sessions.

HINTs are the lightweight alternative to injecting full prompt text.
When context size and structure are uncertain (dynamic escalation rounds,
varying memory recall results, changing step-type catalogs), HINTs provide
compact guidance signals:

    HINT: 2 memory entries found — use recall_memory(query="...") to retrieve.
    HINT: Pipeline round 2: 3/3 steps OK, all goals SAT.
    HINT: 1 custom method registered: solve_problem_plan_build_test.
    HINT: Previous L1 attempt failed — consider a different approach.
    HINT: 3 step types available: specify, implement, evaluate.

Each HINT is one line.  They are assembled into the ``system.hints`` block
by ``MetaPlannerBlockProvider`` and rendered by the Context Layer.

Design principle: HINTs tell MetaPlanner *what is available* or *what
happened*, not *what to do*.  The decision is MetaPlanner's.
"""

from __future__ import annotations

from typing import Any


class MetaHintsBuilder:
    """Build dynamic HINT lines for a MetaPlanner session.

    Usage::

        builder = MetaHintsBuilder()
        builder.add_memory_hint(count=3)
        builder.add_pipeline_hint(round_idx=1, ok=3, total=3, sat=True)
        builder.add_method_hint(count=2, ids=["solve_problem_A", "solve_problem_B"])
        builder.add_step_types_hint(count=4, names=["specify", "implement", "evaluate"])
        builder.add_previous_failure_hint()
        hints = builder.build()

        # hints =
        # HINT: 3 memory entries found — use recall_memory(query="...") to retrieve.
        # HINT: Pipeline round 2: 3/3 steps OK, all goals SAT.
        # HINT: 2 custom methods registered: solve_problem_A, solve_problem_B.
        # HINT: 4 step types available: specify, implement, evaluate, ...
        # HINT: Previous L1 attempt failed — consider a different approach.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []

    # -- Public builders -----------------------------------------------------

    def add_memory_hint(self, count: int) -> None:
        """Add HINT about available memory entries."""
        if count <= 0:
            return
        self._lines.append(
            f"HINT: {count} memory entr{'y' if count == 1 else 'ies'} found "
            f"— use recall_memory(query=\"...\") to retrieve."
        )

    def add_decision_hint(self, count: int) -> None:
        """Add HINT about prior design decisions available for recall.

        Kept as a dedicated HINT line (separate from the memory HINT) so
        the decision store can evolve its own lifecycle / compression
        policy independent of procedural learning.
        """
        if count <= 0:
            return
        self._lines.append(
            f"HINT: {count} prior design decision(s) recorded — use "
            f"recall_memory(query=\"...\") to review the previous approach."
        )

    def add_pipeline_hint(
        self,
        *,
        round_idx: int,
        ok: int,
        total: int,
        sat: bool,
    ) -> None:
        """Add HINT about the most recent pipeline execution."""
        sat_str = "all goals SAT" if sat else "goals UNSAT — review diagnosis"
        self._lines.append(
            f"HINT: Pipeline round {round_idx + 1}: "
            f"{ok}/{total} steps OK, {sat_str}."
        )

    def add_method_hint(self, count: int, ids: list[str] | None = None) -> None:
        """Add HINT about registered custom methods."""
        if count <= 0:
            return
        id_str = ""
        if ids:
            shown = ids[:3]
            id_str = f": {', '.join(shown)}"
            if len(ids) > 3:
                id_str += f", ... (+{len(ids) - 3} more)"
        self._lines.append(
            f"HINT: {count} custom method(s) registered{id_str}."
        )

    def add_step_types_hint(self, count: int, names: list[str] | None = None) -> None:
        """Add HINT about available step types."""
        if count <= 0:
            return
        name_str = ""
        if names:
            shown = names[:5]
            name_str = f": {', '.join(shown)}"
            if len(names) > 5:
                name_str += f", ... (+{len(names) - 5} more)"
        self._lines.append(
            f"HINT: {count} step type(s) available{name_str}."
        )

    def add_ppf_hint(self, *, has_ppf: bool, plan_count: int = 0) -> None:
        """Add a HINT about PPF / plan recall availability.

        ``has_ppf`` indicates the problem has been understood into PPF;
        ``plan_count`` is the number of retained plan-overwrite snapshots
        available for recall (0 = first round).
        """
        if has_ppf and plan_count > 0:
            self._lines.append(
                "HINT: PPF recorded; "
                f"{plan_count} prior plan snapshot(s) available for recall."
            )
        elif has_ppf:
            self._lines.append(
                "HINT: PPF recorded (first round — no prior plan snapshot)."
            )
        else:
            self._lines.append(
                "HINT: PPF not yet recorded — understand the problem first."
            )

    def add_previous_failure_hint(self) -> None:
        """Add HINT that a previous L1 attempt failed."""
        self._lines.append(
            "HINT: Previous L1 attempt failed — consider a different approach."
        )

    def add_no_methods_hint(self) -> None:
        """Add HINT that no custom methods exist yet."""
        self._lines.append(
            "HINT: No custom methods registered — use define_method to create one."
        )

    def add_recall_empty_hint(self) -> None:
        """Add HINT that recall returned empty (expected on first run)."""
        self._lines.append(
            "HINT: Memory recall returned empty — this is expected on first run."
        )

    def add_custom_hint(self, text: str) -> None:
        """Add a custom HINT line."""
        self._lines.append(f"HINT: {text}")

    # -- Build ---------------------------------------------------------------

    def build(self) -> str:
        """Return the assembled HINTS block as a string.

        Returns empty string when no HINTs were added.
        """
        if not self._lines:
            return ""
        header = "## HINTS"
        return header + "\n" + "\n".join(self._lines)

    def clear(self) -> None:
        """Reset all HINT lines."""
        self._lines.clear()

    # -- Convenience: build from session state -------------------------------

    @classmethod
    def from_session_state(
        cls,
        *,
        memory_count: int = 0,
        round_idx: int = 0,
        step_ok: int = 0,
        step_total: int = 0,
        all_sat: bool = False,
        method_count: int = 0,
        method_ids: list[str] | None = None,
        step_type_count: int = 0,
        step_type_names: list[str] | None = None,
        previous_failed: bool = False,
    ) -> str:
        """Build HINTS from structured session state in one call.

        Convenience for callers that have all state available.
        """
        builder = cls()
        if memory_count > 0:
            builder.add_memory_hint(memory_count)
        else:
            builder.add_recall_empty_hint()

        if step_total > 0:
            builder.add_pipeline_hint(
                round_idx=round_idx, ok=step_ok, total=step_total, sat=all_sat,
            )

        if method_count > 0:
            builder.add_method_hint(method_count, method_ids)
        elif round_idx > 0:
            builder.add_no_methods_hint()

        builder.add_step_types_hint(step_type_count, step_type_names)

        if previous_failed:
            builder.add_previous_failure_hint()

        return builder.build()
