"""File-backed IAuditLog implementation.

Writes audit events to a JSONL file under the session directory.
One event per line; fsync per event for crash safety at step-built
cadence (verified against store.py write cost).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from quro.core.protocols import AuditEvent, IAuditLog


class FileAuditLog:
    """File-backed IAuditLog implementation.

    Writes audit events to a JSONL file under the session directory.
    One event per line; fsync per event for crash safety at step-built
    cadence (verified against store.py write cost).
    """

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self._events.append(event)
        line = json.dumps({
            "round_idx": event.round_idx,
            "kind": event.kind,
            "payload": event.payload,
            "ts": event.ts,
        }, ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def events_for_round(self, round_idx: int) -> list[AuditEvent]:
        return [e for e in self._events if e.round_idx == round_idx]

    def events_by_kind(self, kind: str) -> list[AuditEvent]:
        return [e for e in self._events if e.kind == kind]
