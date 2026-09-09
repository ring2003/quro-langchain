"""RoundController — drives one round's g -> f -> evaluate -> gate lifecycle.

Owns ONE RoundScope. Constructed fresh per round by Session.

Architecture §5 lifecycle:
  OPEN -> BUILD -> EXECUTE -> EVALUATE -> GATE -> CLOSE

Phase 3 deliverable (architecture §3.3).
"""

from __future__ import annotations

import time
from typing import Any

from quro.core.protocols import (
    AuditEvent,
    IAuditLog,
    ICrossRoundHandoff,
    IPipelineController,
    StepBuildResult,
)
from quro.core.round_scope import RoundScope


class PlanGate:
    """Per-round plan/stop gate."""

    def __init__(self) -> None:
        self.plan_text: str = ""
        self.stop: bool = False
        self.conclusion: str = ""


class RoundController:
    """Drives one round's g -> f -> evaluate -> gate lifecycle.

    Owns ONE RoundScope. Constructed fresh per round by Session.

    Architecture §5 lifecycle:
      OPEN -> BUILD -> EXECUTE -> EVALUATE -> GATE -> CLOSE
    """

    def __init__(
        self,
        round_idx: int,
        builder: Any,  # IStepBuilder
        controller: Any,  # IPipelineController
        gate: PlanGate,
        audit_log: IAuditLog,
        handoff: ICrossRoundHandoff | None = None,
    ) -> None:
        self._round_idx = round_idx
        self._scope = RoundScope(round_idx=round_idx)
        self._builder = builder
        self._controller = controller
        self._gate = gate
        self._audit_log = audit_log
        self._handoff = handoff
        # Latest executed pipeline result, carried into the handoff on close().
        self._last_result: Any = None
        self._phase = "OPEN"

    def open(self) -> None:
        """Phase: OPEN — emit round_open audit event."""
        self._audit_log.emit(AuditEvent(
            round_idx=self._round_idx,
            kind="round_open",
            payload={"round_idx": self._round_idx},
            ts=time.time(),
        ))
        self._phase = "BUILD"

    def build_step(self, **kwargs: Any) -> StepBuildResult:
        """Phase: BUILD — delegate to IStepBuilder, add to scope."""
        result = self._builder.create(**kwargs)
        self._scope.add(result)
        return result

    def execute(self) -> Any:
        """Phase: EXECUTE — run pipeline via IPipelineController.

        Records the executed result so ``close()`` can carry the round's
        artifacts into the cross-round handoff.
        """
        self._phase = "EXECUTE"
        result = self._controller.execute(self._scope)
        self._scope.executed = True
        self._last_result = result
        # Carry immediately so a get_artifact in the SAME session (right after
        # run_custom_pipeline) already resolves; close() re-carries (idempotent).
        if self._handoff is not None:
            self._carry_artifacts()
        return result

    def close(self) -> None:
        """Phase: CLOSE — carry executed steps + artifacts, emit warnings, discard scope."""
        # Carry executed steps and their artifacts to the cross-round handoff.
        if self._scope.executed and self._handoff is not None:
            for step_id, result in self._scope.pending_steps.items():
                self._handoff.carry_step_ref(self._round_idx, step_id, result)
            self._carry_artifacts()

        # Warn about unexecuted steps.
        if not self._scope.executed and self._scope.pending_steps:
            for step_id in self._scope.pending_steps:
                self._audit_log.emit(AuditEvent(
                    round_idx=self._round_idx,
                    kind="step_carried_unexecuted",
                    payload={"step_id": step_id},
                    ts=time.time(),
                ))

        self._audit_log.emit(AuditEvent(
            round_idx=self._round_idx,
            kind="gate_closed",
            payload={"plan_text": self._gate.plan_text, "stop": self._gate.stop},
            ts=time.time(),
        ))
        self._phase = "CLOSE"

    def _carry_artifacts(self) -> None:
        """Carry the executed round's artifacts into the cross-round handoff."""
        from quro.core.cross_round_handoff import carry_result_artifacts

        carry_result_artifacts(self._handoff, self._round_idx, self._last_result)

    @property
    def scope(self) -> RoundScope:
        return self._scope

    @property
    def round_idx(self) -> int:
        return self._round_idx

    @property
    def gate(self) -> PlanGate:
        return self._gate

    @property
    def phase(self) -> str:
        return self._phase
