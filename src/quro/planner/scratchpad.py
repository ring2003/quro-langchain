"""Scratchpad — the meta-planner's stateful planning surface.

The scratchpad holds the four data concepts fixed by
``meta-plan-prompt-and-ppf.md`` §2:

- **PPF** (``PPF``) — a pure-static problem description the agent records:
  clarified intent + goals + constraints + exploration skeleton.  No rounds.
- **plan** (``PlanSnapshot`` via the journal) — the agent's *single* planning
  artifact, overwrite-only.  Every resubmission replaces the previous version;
  each overwrite is recorded as an immutable full-text snapshot.
- **rounds** (``RoundRecord``) — the framework-owned read-only execution
  trace (``actual`` = status/artifacts), never submitted by the agent.
- **audit events** — plan overwrites and escalation records, persisted in
  ``IMetaPlannerSessionJournal``.

The scratchpad is **not** part of the pipeline and its lifecycle does not
depend on the pipeline existing.  It does not drive the pipeline and the
pipeline does not drive it (``continuation-verdict-semantics.md`` §3.1).

Stopping is free: the agent stops by deciding to stop.  Continuing requires a
plan (``meta-plan-prompt-and-ppf.md`` §3): the ``submit_plan`` gate must be
passed before the next round executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quro.planner.session_journal import (
    IMetaPlannerSessionJournal,
    InMemoryMetaPlannerSessionJournal,
    PlanSnapshot,
)


# ============================================================================
# PPF — ProblemPlanFormat (pure static, no rounds)
# ============================================================================


@dataclass
class PPF:
    """The standard problem-description format (static fields only).

    Disciplines how the agent records its problem interpretation and
    exploration skeleton.  Does **not** contain rounds (rounds are a separate
    concept — see ``RoundRecord``).
    """

    problem: str = ""
    """The clarified user intent (the user input, understood)."""

    goals: list[str] = field(default_factory=list)
    """Goal facts that must all be satisfied."""

    constraints: list[str] = field(default_factory=list)
    """Problem constraints."""

    plan: list[str] = field(default_factory=list)
    """The exploration skeleton — branches, not a step list."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "plan": list(self.plan),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PPF":
        return cls(
            problem=d.get("problem", ""),
            goals=list(d.get("goals", [])),
            constraints=list(d.get("constraints", [])),
            plan=list(d.get("plan", [])),
        )


# ============================================================================
# RoundRecord — framework-owned read-only execution trace
# ============================================================================


@dataclass
class RoundRecord:
    """One round's **actual** outcome, written by the framework (read-only).

    The former ``planned`` half of a round was promoted into the standalone
    ``plan`` block (the scratchpad's overwrite-only artifact).  Rounds keep
    only ``actual``: status facts + artifacts.
    """

    round_idx: int
    goal_status: dict[str, Any] = field(default_factory=dict)
    """Outcome facts (status, artifacts) as computed by ``evaluate``."""

    artifacts: list[str] = field(default_factory=list)
    """Artifact ids produced by this round."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "goal_status": dict(self.goal_status),
            "artifacts": list(self.artifacts),
        }


# ============================================================================
# PlanGate — the per-round structured plan/stop gate
# ============================================================================


@dataclass
class PlanGate:
    """The per-round structured plan/stop gate (framework-side).

    ``submit_plan`` / ``stop`` tools write this during the agent's round;
    the loop reads it afterwards.  Encodes the boundary "stop is free; to
    continue, submit a plan" (``meta-plan-prompt-and-ppf.md`` §3).
    """

    plan_text: str | None = None
    """The plan the agent submitted this round (None = not submitted)."""

    stop: bool = False
    """True when the agent declared stop (free stop)."""

    conclusion: str | None = None
    """Optional conclusion text the agent wrote when stopping."""


# ============================================================================
# Scratchpad
# ============================================================================


class Scratchpad:
    """The agent's external problem-solving surface (stateful sidecar).

    Reads and writes the PPF + the single overwrite-only ``plan`` across
    rounds, and reads the framework-owned ``rounds`` trace.  It carries no
    forced terminal state: the agent writes, erases, rewrites, and stops
    because it decides to stop.

    Args:
        journal: The audit journal for plan-overwrite snapshots.  Defaults to
            an in-memory journal when omitted.
    """

    def __init__(
        self,
        journal: IMetaPlannerSessionJournal | None = None,
        run_id: str | None = None,
    ) -> None:
        self._journal = journal or InMemoryMetaPlannerSessionJournal()
        self._run_id = run_id
        self._ppf: PPF | None = None
        self._plan: str | None = None
        self._rounds: list[RoundRecord] = []
        self._conclusion: str | None = None

    # ------------------------------------------------------------- lifecycle
    def open(self, ppf: PPF) -> None:
        """Open the scratchpad with the understood problem (PPF)."""
        self._ppf = ppf
        self._plan = None
        self._rounds.clear()
        self._conclusion = None

    # ------------------------------------------------------------------ plan
    @property
    def plan(self) -> str | None:
        """The current (single) plan text, or None if not yet submitted."""
        return self._plan

    def submit_plan(self, text: str, *, round_idx: int = -1) -> None:
        """Overwrite the plan with *text* and record an immutable snapshot.

        Overwriting **is** the "veto a branch / re-aim" act.  A verbatim
        resubmission is valid — it is the explicit statement "keep the
        current plan".
        """
        self._plan = text
        self._journal.record_plan(
            PlanSnapshot(round_idx=round_idx, text=text, run_id=self._run_id)
        )

    def plan_history(self) -> list[PlanSnapshot]:
        """Return retained plan-overwrite snapshots, oldest first."""
        return self._journal.plan_history()

    # ----------------------------------------------------------------- rounds
    def record_round(self, record: RoundRecord) -> None:
        """Append a framework-computed round outcome (read-only trace)."""
        self._rounds.append(record)

    def rounds(self) -> list[RoundRecord]:
        """Return the framework-owned round trace (read-only)."""
        return list(self._rounds)

    # ------------------------------------------------------------- conclusion
    def conclude(self, text: str) -> None:
        """Write the user-facing conclusion statement.

        This is semantic output (a statement for the user), not a control
        signal: the framework never reads it to force a stop.
        """
        self._conclusion = text

    @property
    def conclusion(self) -> str | None:
        return self._conclusion

    # ------------------------------------------------------------------ read
    @property
    def ppf(self) -> PPF | None:
        return self._ppf
