"""Job key + job index — the self-describing job layer (implementation plan §2 stage 2).

A job is the top of the resource hierarchy::

    job://{domain}/{problem_name}/{goal_uuid}

- ``domain`` — the domain name (``codebase_research``, …).
- ``problem_name`` — a user-required RFC-style slug
  ``[a-z0-9][a-z0-9-]{0,62}`` (lowercase, hyphens, path-safe).  Fail-fast on
  validation — it is a path component.
- ``goal_uuid`` — ``sha1(sorted(goal_facts))[:16]``.  Goal facts changed → new
  uuid → new job sub-directory → old artifacts stay findable under the old
  goal.  Deterministic, no migration.

``JobIndex`` reads/upserts ``jobs/{domain}/{problem_name}/{goal_uuid}/
index.json`` — the job's self-description + session list.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# RFC-style slug: lowercase letters/digits, hyphens as internal separators.
PROBLEM_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

INDEX_VERSION = 1


class InvalidJobDescriptor(ValueError):
    """Raised when a job descriptor does not parse or fails validation."""


@dataclass(frozen=True)
class JobKey:
    """The stable identity of a job: domain + problem slug + goal uuid."""

    domain: str
    problem_name: str
    goal_uuid: str

    def __post_init__(self) -> None:
        if not self.domain or "/" in self.domain or "\\" in self.domain:
            raise InvalidJobDescriptor(
                f"invalid domain {self.domain!r}: must be a non-empty path component"
            )
        validate_problem_name(self.problem_name)
        if not self.goal_uuid:
            raise InvalidJobDescriptor("goal_uuid must be non-empty")

    def descriptor(self) -> str:
        return f"job://{self.domain}/{self.problem_name}/{self.goal_uuid}"

    def __str__(self) -> str:
        return self.descriptor()


def validate_problem_name(name: str) -> str:
    """Validate and return *name* as an RFC-style slug (fail-fast).

    Raises :class:`InvalidJobDescriptor` when *name* does not match
    ``[a-z0-9][a-z0-9-]{0,62}``.
    """
    if not isinstance(name, str) or not PROBLEM_NAME_RE.match(name):
        raise InvalidJobDescriptor(
            f"invalid problem_name {name!r}: must match [a-z0-9][a-z0-9-]{{0,62}} "
            "(lowercase, digits, internal hyphens)"
        )
    return name


def parse_job_descriptor(s: str) -> JobKey:
    """Parse ``job://{domain}/{problem_name}/{goal_uuid}`` into a ``JobKey``.

    Raises :class:`InvalidJobDescriptor` on bad syntax or a failing slug.
    """
    if not isinstance(s, str):
        raise InvalidJobDescriptor(
            f"job descriptor must be a string, got {type(s).__name__}"
        )
    if not s.startswith("job://"):
        raise InvalidJobDescriptor(f"invalid job descriptor {s!r}: missing 'job://'")
    rest = s[len("job://"):]
    parts = rest.split("/")
    if len(parts) != 3 or any(not p for p in parts):
        raise InvalidJobDescriptor(
            f"invalid job descriptor {s!r}: expected job://{{domain}}/"
            "{problem_name}/{goal_uuid}"
        )
    return JobKey(parts[0], parts[1], parts[2])


def goal_uuid_from_facts(goal_facts: list[str] | tuple[str, ...]) -> str:
    """Compute the deterministic goal uuid: ``sha1(sorted(goal_facts))[:16]``.

    Order-insensitive: the same set of goal facts yields the same uuid
    regardless of order; any change to the facts yields a new uuid.
    """
    canonical = json.dumps(
        sorted(str(f) for f in goal_facts), ensure_ascii=True,
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


class JobIndex:
    """Read/upsert the job index under ``{root_dir}/jobs/{domain}/{problem}/{goal}/``.

    The index is self-describing bookkeeping: it lists the job's sessions with
    ``created_at`` and a one-line progress summary, and infers nothing.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    # -- layout -------------------------------------------------------------

    def job_dir(self, job: JobKey) -> Path:
        return self._root / "jobs" / job.domain / job.problem_name / job.goal_uuid

    def _index_path(self, job: JobKey) -> Path:
        return self.job_dir(job) / "index.json"

    # -- API ----------------------------------------------------------------

    def read(self, job: JobKey) -> dict[str, Any] | None:
        """Return the job index dict, or None when it does not exist."""
        p = self._index_path(job)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def list_sessions(self, job: JobKey) -> list[dict[str, Any]]:
        """Return the sessions recorded under *job* (empty when unknown)."""
        index = self.read(job)
        if not index:
            return []
        sessions = index.get("sessions", [])
        return [s for s in sessions if isinstance(s, dict)]

    def upsert(
        self,
        job: JobKey,
        session_id: str,
        *,
        summary: str = "",
        created_at: float | None = None,
    ) -> dict[str, Any]:
        """Record/update *session_id* under *job* and return the new index.

        An existing session keeps its original ``created_at`` and only its
        ``summary`` is refreshed; a new session is appended with a fresh
        ``created_at``.
        """
        index = self.read(job)
        sessions: list[dict[str, Any]] = []
        if index:
            sessions = [s for s in index.get("sessions", []) if isinstance(s, dict)]
        else:
            index = {
                "version": INDEX_VERSION,
                "job": {
                    "domain": job.domain,
                    "problem_name": job.problem_name,
                    "goal_uuid": job.goal_uuid,
                },
                "sessions": [],
            }

        now = created_at if created_at is not None else time.time()
        for s in sessions:
            if s.get("session_id") == session_id:
                if summary:
                    s["summary"] = summary
                self._write(job, index)
                return index

        sessions.append({
            "session_id": session_id,
            "created_at": now,
            "summary": summary,
        })
        index["sessions"] = sessions
        self._write(job, index)
        return index

    # -- helpers ------------------------------------------------------------

    def _write(self, job: JobKey, index: dict[str, Any]) -> None:
        p = self._index_path(job)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, p)
