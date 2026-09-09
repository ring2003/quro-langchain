"""Runtime session ledger — the application-layer resume unit (runtime-session.md §4-9).

The ledger is self-describing bookkeeping only: it records what exists and
where, and infers nothing.  It lives at::

    jobs/{domain}/{problem_name}/{goal_uuid}/sessions/{session_id}/index.json

Schema (runtime-session.md §4): ``version``, ``session_id``, ``job``,
``current_round``, ``current_step``, ``rounds[]``, ``extensions``.

Forward compatibility (three tiers, §7) is enforced by **round-trip
losslessness**: ``load`` returns the whole dict (including unknown fields) and
``save`` writes the whole dict back unchanged.  The framework never parses or
drops a field it does not understand — it skips-and-preserves.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from quro.core.resources.job import JobIndex, JobKey

LEDGER_VERSION = 1

_SESSION_ID_RE = re.compile(r"^s(\d+)$")


class RuntimeSessionLedger:
    """Read/write ``session/index.json`` and allocate stable session ids.

    Args:
        root_dir: The state root — session ledgers live under
            ``{root_dir}/jobs/{domain}/{problem}/{goal}/sessions/{id}/``.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._jobs = JobIndex(root_dir)

    # -- layout -------------------------------------------------------------

    def session_dir(self, job: JobKey, session_id: str) -> Path:
        return self._jobs.job_dir(job) / "sessions" / session_id

    def ledger_path(self, job: JobKey, session_id: str) -> Path:
        return self.session_dir(job, session_id) / "index.json"

    # -- session id allocation ---------------------------------------------

    def allocate_session_id(self, job: JobKey) -> str:
        """Allocate the next incrementing session id under *job* (``s1``, ``s2``…)."""
        existing = self._jobs.list_sessions(job)
        highest = 0
        for s in existing:
            m = _SESSION_ID_RE.match(str(s.get("session_id", "")))
            if m:
                highest = max(highest, int(m.group(1)))
        return f"s{highest + 1}"

    def latest_session_id(self, job: JobKey) -> str | None:
        """Return the most recently recorded session id under *job*, or None."""
        existing = self._jobs.list_sessions(job)
        if not existing:
            return None
        return str(existing[-1].get("session_id") or "") or None

    def create_session(
        self,
        job: JobKey,
        *,
        resume: bool = False,
        summary: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Create (or resume) a session under *job*.

        ``resume=True`` reuses the latest existing session id (or allocates a
        new one when the job has none); ``resume=False`` always allocates a
        fresh incrementing id.  Writes the ledger and records the session in
        the job index.  Returns ``(session_id, ledger)``.
        """
        if resume:
            session_id = self.latest_session_id(job) or self.allocate_session_id(job)
        else:
            session_id = self.allocate_session_id(job)
        ledger = self.load_or_create(job, session_id)
        self.save(job, ledger)
        self._jobs.upsert(job, session_id, summary=summary)
        return session_id, ledger

    # -- ledger schema ------------------------------------------------------

    @staticmethod
    def new_ledger(job: JobKey, session_id: str) -> dict[str, Any]:
        """A fresh ledger with the tier-1 core fields populated."""
        return {
            "version": LEDGER_VERSION,
            "session_id": session_id,
            "job": {
                "domain": job.domain,
                "problem_name": job.problem_name,
                "goal_uuid": job.goal_uuid,
            },
            "current_round": 0,
            "current_step": "",
            "rounds": [],
            "extensions": {},
        }

    def load(self, job: JobKey, session_id: str) -> dict[str, Any] | None:
        """Load the ledger dict verbatim (unknown fields preserved), or None."""
        p = self.ledger_path(job, session_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def load_or_create(self, job: JobKey, session_id: str) -> dict[str, Any]:
        """Load an existing ledger, or create a fresh one (not yet written)."""
        return self.load(job, session_id) or self.new_ledger(job, session_id)

    def save(self, job: JobKey, ledger: dict[str, Any]) -> None:
        """Write *ledger* back verbatim — unknown fields are never dropped."""
        p = self.ledger_path(job, str(ledger.get("session_id") or ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    def update(
        self,
        job: JobKey,
        session_id: str,
        **fields: Any,
    ) -> dict[str, Any]:
        """Update known fields in-place and write back (unknown fields intact)."""
        ledger = self.load_or_create(job, session_id)
        for key, value in fields.items():
            ledger[key] = value
        self.save(job, ledger)
        return ledger
