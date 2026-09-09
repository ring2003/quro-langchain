"""Bounded cross-round session journal for MetaPlanner rounds.

Records three audit event kinds:

- ``EscalationEntry`` — one MetaPlanner repair round's record, kept for
  cross-round traceability.
- ``PlanSnapshot`` — the immutable full-text snapshot of every ``plan``
  overwrite (the agent's "veto a branch / re-aim" act).  This is the only
  historical record of past planning intent — rounds keep only outcomes.
- ``SteeringRoundEntry`` — one steering round's record for audit and replay.

Design principle (blueprint): consumers depend on the ABC protocol, never on
a concrete backend.

Verdicts are gone (``continuation-verdict-semantics.md`` §5.4); this journal
no longer carries any SAT/UNSAT/CONTINUE signal.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from quro.core.protocols import AuditEvent


@dataclass
class EscalationEntry:
    """One escalation's record for cross-round traceability."""

    round_idx: int
    escalation_mode: str
    diagnosis_cause: str = ""
    unsat_facts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    result: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "escalation_mode": self.escalation_mode,
            "diagnosis_cause": self.diagnosis_cause,
            "unsat_facts": list(self.unsat_facts),
            "actions": list(self.actions),
            "result": self.result,
            "run_id": self.run_id,
        }

    def render(self) -> str:
        """Compact one-line render for prompt injection."""
        bits: list[str] = [f"Round {self.round_idx + 1}", self.escalation_mode]
        if self.result:
            bits.append(f"result={self.result}")
        if self.actions:
            bits.append(f"actions={','.join(self.actions[:5])}")
        if self.unsat_facts:
            bits.append(f"unsat={','.join(self.unsat_facts[:5])}")
        return " | ".join(bits)


@dataclass
class PlanSnapshot:
    """An immutable full-text snapshot of a ``plan`` overwrite.

    LLM rewrites are not diffable; each resubmission stores the full text so
    past planning intent survives (rounds keep only outcomes).
    """

    round_idx: int
    text: str
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "text": self.text,
            "run_id": self.run_id,
        }


@dataclass
class SteeringRoundEntry:
    """One steering cycle's record for audit and replay (Steering-Semantics.md §8).

    Captures the frontier (WHERE the cycle explored), raw LLM output, parsed
    NextInstruction, state view snapshot, gate hint, and exit reason for each cycle.
    """

    round_index: int
    objective: str  # The frontier selected for this cycle
    raw_output: str = ""
    next_instruction: Any = None
    state_view_snapshot: str = ""
    gate_hint: str = ""
    exit_reason: str = ""  # "final_round" | "budget_exhausted"
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "objective": self.objective,
            "raw_output": self.raw_output,
            "next_instruction": (
                self.next_instruction.to_dict()
                if hasattr(self.next_instruction, "to_dict")
                else str(self.next_instruction) if self.next_instruction else None
            ),
            "state_view_snapshot": self.state_view_snapshot,
            "gate_hint": self.gate_hint,
            "exit_reason": self.exit_reason,
            "run_id": self.run_id,
        }

    def render(self) -> str:
        """Compact one-line render for prompt injection."""
        bits: list[str] = [f"Cycle {self.round_index}", f"frontier={self.objective[:40]}"]
        if self.exit_reason:
            bits.append(f"exit={self.exit_reason}")
        return " | ".join(bits)


class IMetaPlannerSessionJournal(ABC):
    """Bounded, structured record of MetaPlanner rounds."""

    @abstractmethod
    def record(self, entry: EscalationEntry) -> None:
        """Append one escalation entry (bounded eviction)."""
        ...

    @abstractmethod
    def record_plan(self, snapshot: PlanSnapshot) -> None:
        """Append one plan-overwrite snapshot (bounded eviction)."""
        ...

    @abstractmethod
    def record_steering_round(self, entry: SteeringRoundEntry) -> None:
        """Append one steering round entry (bounded eviction)."""
        ...

    @abstractmethod
    def recent(self, limit: int = 3) -> list[EscalationEntry]:
        """Return the most recent escalation entries, newest last."""
        ...

    @abstractmethod
    def plan_history(self) -> list[PlanSnapshot]:
        """Return all retained plan-overwrite snapshots, oldest first."""
        ...

    @abstractmethod
    def steering_history(self, limit: int = 10) -> list[SteeringRoundEntry]:
        """Return the most recent steering round entries, oldest first."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Drop all entries (new run)."""
        ...


class InMemoryMetaPlannerSessionJournal(IMetaPlannerSessionJournal):
    """Bounded in-memory journal (default implementation).

    When an ``audit_log`` is provided, escalation and plan-overwrite events
    are also emitted to the shared ``IAuditLog`` (Phase 1).
    """

    def __init__(
        self,
        max_entries: int = 10,
        audit_log: Any | None = None,
    ) -> None:
        self._entries: list[EscalationEntry] = []
        self._plan_snapshots: list[PlanSnapshot] = []
        self._steering_rounds: list[SteeringRoundEntry] = []
        self._max_entries = max_entries
        self._audit_log = audit_log

    def record(self, entry: EscalationEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            del self._entries[: len(self._entries) - self._max_entries]
        if self._audit_log is not None:
            self._audit_log.emit(AuditEvent(
                round_idx=entry.round_idx,
                kind="escalation",
                payload=entry.to_dict(),
                ts=_time.time(),
            ))

    def record_plan(self, snapshot: PlanSnapshot) -> None:
        self._plan_snapshots.append(snapshot)
        if len(self._plan_snapshots) > self._max_entries:
            del self._plan_snapshots[: len(self._plan_snapshots) - self._max_entries]
        if self._audit_log is not None:
            self._audit_log.emit(AuditEvent(
                round_idx=snapshot.round_idx,
                kind="plan_overwritten",
                payload=snapshot.to_dict(),
                ts=_time.time(),
            ))

    def record_steering_round(self, entry: SteeringRoundEntry) -> None:
        self._steering_rounds.append(entry)
        if len(self._steering_rounds) > self._max_entries:
            del self._steering_rounds[: len(self._steering_rounds) - self._max_entries]
        if self._audit_log is not None:
            self._audit_log.emit(AuditEvent(
                round_idx=entry.round_index,
                kind="steering_round",
                payload=entry.to_dict(),
                ts=_time.time(),
            ))

    def recent(self, limit: int = 3) -> list[EscalationEntry]:
        return list(self._entries[-limit:])

    def plan_history(self) -> list[PlanSnapshot]:
        return list(self._plan_snapshots)

    def steering_history(self, limit: int = 10) -> list[SteeringRoundEntry]:
        return list(self._steering_rounds[-limit:])

    def clear(self) -> None:
        self._entries.clear()
        self._plan_snapshots.clear()
        self._steering_rounds.clear()
