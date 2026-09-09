"""Resume — replay events after the last snapshot (architecture §5).

A round-boundary snapshot is the replay anchor; resume = replay the events
appended after that snapshot, never replay from zero.  Events are round-scoped
operations over ``DomainState``; ``apply_event`` is the (small, explicit) op
vocabulary.
"""

from __future__ import annotations

import copy
from typing import Any

from quro.core.resources.refs import ResourceRef
from quro.core.resources.store import IResourceStore


def apply_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Apply one domain event to *state* (mutates and returns it).

    Recognised ops (unknown ops are skipped — forward compatible):
        - ``add_artifact``          → ``artifacts`` append
        - ``update_step_status``    → ``steps`` status
        - ``commit_clue``           → ``recovery[step_id].clues`` append
        - ``set_plan``              → ``plan``
        - ``set_interpretation``    → ``interpretation``
        - ``set_phase``             → ``phase`` / ``executing_step_id``
    """
    op = event.get("op")
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return state

    if op == "add_artifact":
        art = payload.get("artifact")
        if isinstance(art, dict) and art.get("artifact_id"):
            state.setdefault("artifacts", []).append(dict(art))
    elif op == "update_step_status":
        for s in state.setdefault("steps", []):
            if isinstance(s, dict) and s.get("step_id") == payload.get("step_id"):
                s["status"] = payload.get("status")
                break
    elif op == "commit_clue":
        clue = payload.get("clue")
        if clue:
            journal = state.setdefault("recovery", {}).setdefault(
                str(payload.get("step_id") or ""), {}
            )
            journal.setdefault("clues", []).append(clue)
    elif op == "set_plan":
        state["plan"] = payload.get("plan", "")
    elif op == "set_interpretation":
        state["interpretation"] = payload.get("text", "")
    elif op == "set_phase":
        if payload.get("phase"):
            state["phase"] = payload["phase"]
        if payload.get("executing_step_id"):
            state["executing_step_id"] = payload["executing_step_id"]
    return state


def replay_events(
    base_state: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay *events* over *base_state*, returning a fresh cumulative state."""
    state = copy.deepcopy(base_state or {})
    for event in events:
        if isinstance(event, dict):
            apply_event(state, event)
    return state


def resume_domain_state(
    store: IResourceStore,
    snapshot_ref: ResourceRef,
    *,
    event_ref: ResourceRef | None = None,
) -> dict[str, Any]:
    """Resume the ``DomainState`` addressed by *snapshot_ref*.

    The snapshot is the replay anchor; events appended to *event_ref* (default:
    none) are replayed on top.  Returns ``{}`` when no snapshot exists.
    """
    snapshot = store.load_snapshot(snapshot_ref)
    if snapshot is None:
        return {}
    base = snapshot.get("domain_state")
    if not isinstance(base, dict):
        base = {}
    events = store.read_events(event_ref) if event_ref is not None else []
    return replay_events(base, events)
