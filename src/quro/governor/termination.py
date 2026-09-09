"""Default TerminationPolicy — progress-based idle detection.

Replaces the broken ``consecutive_idle_rounds`` counter that incremented
unconditionally in non-``step_spec`` phases (§4.1, §9 of the governance report).
"""

from __future__ import annotations

from typing import Any

from quro.governor.core import (
    ProgressSnapshot,
    TerminationDecision,
    TerminationReason,
)


class TerminationPolicy:
    """Protocol for step termination decisions."""

    def should_terminate(
        self,
        *,
        state: dict[str, Any],
        round_index: int,
        progress: ProgressSnapshot,
        last_decision: TerminationDecision | None,
    ) -> TerminationDecision | None:
        ...


class DefaultTerminationPolicy:
    """Terminate when progress stalls for N consecutive rounds.

    Unlike the old idle counter, progress is measured by *observable
    output changes* (phase transition, step/artifact count change,
    terminal tool invocation), not by a counter that unconditionally
    increments.
    """

    def __init__(self, *, max_idle_rounds: int = 3) -> None:
        self.max_idle_rounds = max_idle_rounds
        self._prev_progress: ProgressSnapshot | None = None
        self._idle_count: int = 0

    def should_terminate(
        self,
        *,
        state: dict[str, Any],
        round_index: int,
        progress: ProgressSnapshot,
        last_decision: TerminationDecision | None,
    ) -> TerminationDecision | None:
        # Phase "done" is always a success termination
        if state.get("phase") == "done":
            return TerminationDecision(
                TerminationReason.PHASE_DONE, ok=True,
                detail="Domain phase reached 'done'",
            )

        if self._prev_progress is None:
            self._prev_progress = progress
            return None

        if progress.has_progress(self._prev_progress):
            self._idle_count = 0
        else:
            self._idle_count += 1

        self._prev_progress = progress

        if self._idle_count >= self.max_idle_rounds:
            return TerminationDecision(
                TerminationReason.IDLE_STALL, ok=False,
                detail=f"No output progress for {self._idle_count} consecutive rounds",
            )

        return None
