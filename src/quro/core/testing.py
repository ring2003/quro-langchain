"""Named test doubles for the Pipeline/Step/Context Controller.

Named test doubles (architecture §3.6 -- never silent defaults).
These are for testing only and must never be used in production wiring.
"""

from __future__ import annotations

import time
from typing import Any

from quro.core.protocols import AuditEvent, IAuditLog


class InMemoryAuditLog:
    """Named test double -- never used in production wiring."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)

    def events_for_round(self, round_idx: int) -> list[AuditEvent]:
        return [e for e in self.events if e.round_idx == round_idx]

    def events_by_kind(self, kind: str) -> list[AuditEvent]:
        return [e for e in self.events if e.kind == kind]
