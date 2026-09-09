"""CLI ``show`` view — a pure, stateless, read-only projection (phase 3).

The view renders a problem-status summary at three depths — job, session,
round — addressed by a job descriptor.  It depends on the resource-layer
*protocols* (``IResourceStore`` / ``JobIndex`` / ``RuntimeSessionLedger``),
never a concrete backend, and never imports the quro-thinking kernel.

Every read is a ``ResourceRef``-keyed storage operation.  The projection never
mutates, never materialises payloads, and never triggers a resume: a missing
snapshot / dangling reference renders as an explicit placeholder, never a
crash (weak-reference behaviour, phase-2 ``offload.py``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quro.core.resources import (
    IResourceStore,
    JobIndex,
    JobKey,
    ResourceRef,
    RuntimeSessionLedger,
)

# Round-boundary snapshot sub-paths written by
# ``steps/checkpoint_hook.py::RoundCheckpoint`` (round://{idx}/start|end).
_ROUND_START = ("start",)
_ROUND_END = ("end",)

# Goal preview length for the per-round one-line summary.
_GOAL_PREVIEW = 60


class JobNotFoundError(LookupError):
    """Raised when a job descriptor addresses no known job."""


class SessionNotFoundError(LookupError):
    """Raised when a session id is not recorded under a job."""


class CLIView:
    """Stateless renderer for ``quro show`` (job / session / round depths)."""

    # -- job level ---------------------------------------------------------

    def render_job(self, job: JobKey, index: JobIndex) -> str:
        """Render the job summary: its sessions with id + created_at + summary."""
        if index.read(job) is None:
            raise JobNotFoundError(job.descriptor())
        sessions = index.list_sessions(job)
        lines = [f"Job: {job.descriptor()}", ""]
        if not sessions:
            lines.append("Sessions: (none)")
            return "\n".join(lines)
        lines.append("Sessions:")
        for session in sessions:
            sid = session.get("session_id", "?")
            created = _format_created_at(session.get("created_at"))
            summary = str(session.get("summary") or "")
            line = f"  {sid}"
            if created:
                line += f"  {created}"
            if summary:
                line += f"  {summary}"
            lines.append(line)
        return "\n".join(lines)

    # -- session level -----------------------------------------------------

    def render_session(
        self,
        job: JobKey,
        session_id: str,
        ledger: RuntimeSessionLedger,
        store: IResourceStore,
    ) -> str:
        """Render the session summary: resume point + per-round goal_status/goal.

        Round goal_status is read from ``RoundCheckpoint`` snapshots, not
        ``ledger.rounds[]`` (which is currently unpopulated — gap G2); the
        snapshots are the authoritative round record.
        """
        data = ledger.load(job, session_id)
        if data is None:
            raise SessionNotFoundError(f"{job.descriptor()} --session {session_id}")

        current_round = data.get("current_round", 0)
        current_step = data.get("current_step", "")
        lines = [
            f"Session: {session_id}",
            f"Job: {job.descriptor()}",
            f"current_round: {current_round}",
            f"current_step: {current_step or '(none)'}",
            "",
        ]

        rounds = _discover_rounds(store, data)
        if not rounds:
            lines.append("Rounds: (none)")
            return "\n".join(lines)

        lines.append("Rounds:")
        for idx in sorted(rounds):
            start, end = rounds[idx]
            if start is None and end is None:
                lines.append(f"  round {idx}  (missing snapshot)")
                continue
            goal_status = end.get("goal_status") if end else None
            goal = (
                (end.get("goal") if end else None)
                or (start.get("goal") if start else None)
                or ""
            )
            status_s = str(goal_status) if goal_status else "(unknown)"
            goal_s = _truncate(str(goal), _GOAL_PREVIEW) if goal else ""
            line = f"  round {idx}  goal_status={status_s}"
            if goal_s:
                line += f"  goal={goal_s}"
            lines.append(line)
        return "\n".join(lines)

    # -- round level -------------------------------------------------------

    def render_round(self, store: IResourceStore, round_idx: int) -> str:
        """Render one round: steps + artifacts from its ``domain_state``."""
        start = _load_round_snapshot(store, round_idx, _ROUND_START)
        end = _load_round_snapshot(store, round_idx, _ROUND_END)

        if start is None and end is None:
            return f"Round: {round_idx}\n(missing snapshot)"

        lines = [f"Round: {round_idx}"]
        goal = (
            (end.get("goal") if end else None)
            or (start.get("goal") if start else None)
        )
        goal_status = end.get("goal_status") if end else None
        if goal:
            lines.append(f"goal: {_truncate(str(goal), _GOAL_PREVIEW)}")
        if goal_status:
            lines.append(f"goal_status: {goal_status}")

        state = _round_domain_state(start, end)
        lines.append(f"phase: {state.get('phase') or '(unknown)'}")
        lines.append(f"executing_step_id: {state.get('executing_step_id') or '(none)'}")

        lines.append("")
        lines.append("Steps:")
        steps = [s for s in state.get("steps") or [] if isinstance(s, dict)]
        if not steps:
            lines.append("  (none)")
        for step in steps:
            lines.append(
                f"  {step.get('step_id', '?')}  {step.get('status', '(unknown)')}"
            )

        lines.append("")
        lines.append("Artifacts:")
        artifacts = [a for a in state.get("artifacts") or [] if isinstance(a, dict)]
        if not artifacts:
            lines.append("  (none)")
        for artifact in artifacts:
            parts = [f"  {artifact.get('artifact_id', '?')}"]
            kind = artifact.get("kind")
            if kind:
                parts.append(f"kind={kind}")
            summary = artifact.get("summary")
            if summary:
                parts.append(str(summary))
            lines.append("  ".join(parts))
        return "\n".join(lines)


# -- helpers ---------------------------------------------------------------


def _load_round_snapshot(
    store: IResourceStore, round_idx: int, which: tuple[str, ...],
) -> dict[str, Any] | None:
    return store.load_snapshot(ResourceRef("round", str(round_idx), which))


def _round_domain_state(
    start: dict[str, Any] | None, end: dict[str, Any] | None,
) -> dict[str, Any]:
    """The round's ``domain_state``: prefer the end snapshot, then start."""
    for snapshot in (end, start):
        if snapshot and isinstance(snapshot.get("domain_state"), dict):
            return snapshot["domain_state"]
    return {}


def _discover_rounds(
    store: IResourceStore, ledger: dict[str, Any],
) -> dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Enumerate the round indexes recorded by the session.

    The ledger's ``rounds[]`` list is currently unpopulated (gap G2), so the
    round list is derived by probing ``round://{idx}/start|end`` snapshots for
    every index up to ``current_round`` (plus any ``round_idx`` already
    recorded in ``rounds[]``, for forward compatibility).  A missing snapshot
    yields ``None`` and renders as a placeholder upstream.
    """
    current_round = ledger.get("current_round")
    if not isinstance(current_round, int) or current_round < 0:
        current_round = 0

    indexes: set[int] = set(range(current_round + 1))
    for entry in ledger.get("rounds") or []:
        if isinstance(entry, dict) and isinstance(entry.get("round_idx"), int):
            indexes.add(entry["round_idx"])

    rounds: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    for idx in sorted(indexes):
        rounds[idx] = (
            _load_round_snapshot(store, idx, _ROUND_START),
            _load_round_snapshot(store, idx, _ROUND_END),
        )
    return rounds


def _format_created_at(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
