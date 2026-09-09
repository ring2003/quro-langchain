"""InMemoryWorkingMemory — L0 in-session accumulation (Phase 1).

Implements ``IWorkingMemory`` as a simple dict-based accumulator.
Carries UNSAT_facts and progress summaries across rounds within a
single agent-driven loop run.

This is the Phase 1 implementation (blueprint §10).  It has no
external dependencies and runs entirely in-process.
"""

from __future__ import annotations

from typing import Any

from quro.memory.protocols import IWorkingMemory


class InMemoryWorkingMemory(IWorkingMemory):
    """Dict-based working memory for in-session state injection.

    Usage::

        wm = InMemoryWorkingMemory()
        wm.update(0, {"unsat_facts": ["a"], "completed_steps": 3})
        wm.snapshot()       # → {"rounds": [...], "latest": {...}}
        wm.previous_unsat_facts()  # → ["a"]
    """

    def __init__(self) -> None:
        self._rounds: list[dict[str, Any]] = []
        self._latest_unsat: list[str] | None = None

    # ----------------------------------------------------------------- update

    def update(self, round_index: int, summary: dict[str, Any]) -> None:
        """Record a round's compacted summary.

        Only the *latest* unsat_facts are tracked; full history is
        available via ``snapshot()``.
        """
        entry = {"round": round_index, **summary}
        self._rounds.append(entry)
        unsat = summary.get("unsat_facts")
        if isinstance(unsat, list):
            self._latest_unsat = list(unsat)

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, Any]:
        """Return the current working memory state.

        Suitable for injection into the planner's next-round context as a
        compact prompt prefix.
        """
        latest = self._rounds[-1] if self._rounds else {}
        return {
            "rounds": list(self._rounds),
            "total_rounds": len(self._rounds),
            "latest": {
                "unsat_facts": latest.get("unsat_facts", []),
                "sat_facts": latest.get("sat_facts", []),
                "disposition": latest.get("disposition", "continue"),
                "cause": latest.get("cause", ""),
                "completed_steps": latest.get("completed_steps", 0),
            },
        }

    # ----------------------------------------------------- previous_unsat

    def previous_unsat_facts(self) -> list[str] | None:
        """UNSAT_facts from the most recent round, or None."""
        return self._latest_unsat

    # ----------------------------------------------------------------- reset

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._rounds.clear()
        self._latest_unsat = None

    def get_l1_context(self) -> dict[str, Any]:
        """Return a lightweight context summary for L1 MetaPlanner injection.

        Rule-based extraction from accumulated rounds — no LLM call.
        """
        return {
            "rounds": list(self._rounds),
            "total_rounds": len(self._rounds),
            "latest_unsat_facts": self._latest_unsat or [],
            "disposition_history": [
                r.get("disposition", "?") for r in self._rounds
            ],
        }
