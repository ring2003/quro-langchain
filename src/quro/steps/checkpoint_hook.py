"""Checkpointing — the application-layer resume mechanic (runtime-session.md §5).

Three checkpoint layers, two mechanisms:

- **step pre** — :class:`CheckpointHook` (a :class:`StepHook`, first entry of
  ``pre_hooks``): a full snapshot of the current ``DomainState`` + the step's
  stable reasoning-session reference, taken *before* the step executes.  The
  step checkpoint is the resume point for "hack a step without breaking the
  round".
- **round start / round end** — :class:`RoundCheckpoint`, called **directly**
  (not via hooks) by the meta-loop at the round boundary: round goal + input
  ``DomainState`` at start, goal_status + output ``DomainState`` at end.

Step-level checkpoints are **full snapshots** (locked decision 27): simple and
state-complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quro.core.domain.step_config import StepConfig
from quro.core.domain.step_type import StepTypeCatalog
from quro.core.resources import (
    IResourceStore,
    JobKey,
    ResourceRef,
    RuntimeSessionLedger,
)

# Metadata keys the hook leaves for the executor (adapter).
REASONING_SESSION_ID_KEY = "reasoning_session_id"
REASONING_STORAGE_DIR_KEY = "reasoning_storage_dir"


def accumulate_state(results: dict[str, Any]) -> dict[str, Any]:
    """Merge the step results' states into one cumulative ``DomainState``.

    Deterministic merge across the ``steps`` / ``artifacts`` / ``recovery``
    collections (deduplicated by id) and the scalar ``plan`` /
    ``interpretation`` / ``phase`` / ``executing_step_id`` fields (last
    non-empty wins).  ``results`` values may be ``StepResult`` objects or
    plain dicts.
    """
    steps: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    recovery: dict[str, Any] = {}
    plan = interpretation = phase = executing_step_id = ""

    seen_steps: set[str] = set()
    seen_artifacts: set[str] = set()

    for result in (results or {}).values():
        state = _state_of(result)
        if not state:
            continue
        for s in state.get("steps", []):
            if not isinstance(s, dict) or not s.get("step_id"):
                continue
            sid = str(s["step_id"])
            if sid not in seen_steps:
                seen_steps.add(sid)
                steps.append(dict(s))
        for a in state.get("artifacts", []):
            if not isinstance(a, dict) or not a.get("artifact_id"):
                continue
            aid = str(a["artifact_id"])
            if aid not in seen_artifacts:
                seen_artifacts.add(aid)
                artifacts.append(dict(a))
        for sid, journal in (state.get("recovery") or {}).items():
            if isinstance(journal, dict):
                recovery.setdefault(str(sid), journal)
        if state.get("plan"):
            plan = state["plan"]
        if state.get("interpretation"):
            interpretation = state["interpretation"]
        if state.get("phase"):
            phase = state["phase"]
        if state.get("executing_step_id"):
            executing_step_id = state["executing_step_id"]

    return {
        "steps": steps,
        "artifacts": artifacts,
        "recovery": recovery,
        "plan": plan,
        "interpretation": interpretation,
        "phase": phase,
        "executing_step_id": executing_step_id,
    }


def _state_of(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        state = result.get("state")
    else:
        state = getattr(result, "state", None)
    return state if isinstance(state, dict) else None


def restore_domain_state(store: IResourceStore, step_id: str) -> dict[str, Any] | None:
    """Restore the ``DomainState`` snapshotted at the step-pre checkpoint."""
    snapshot = store.load_snapshot(ResourceRef("step", step_id))
    if not snapshot:
        return None
    state = snapshot.get("domain_state")
    return state if isinstance(state, dict) else None


class CheckpointHook:
    """Step-pre checkpoint: full ``DomainState`` snapshot + reasoning reference.

    Runs as the first ``pre_hooks`` entry so the checkpoint exists *before*
    the step executes (the resume point for a simulated interrupt).
    """

    name = "checkpoint"

    def __init__(
        self,
        store: IResourceStore,
        session_id: str,
        *,
        round_idx: int = 0,
        step_types: StepTypeCatalog | None = None,
        ledger: RuntimeSessionLedger | None = None,
        job: JobKey | None = None,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._round_idx = round_idx
        # Optional: inlines identity_block / executor_hints / features from the
        # StepType into the step_config half (phase 4).  None → those fields
        # stay empty in the snapshot.
        self._step_types = step_types
        # Optional ledger hook-in (R0a-2): when set, on_pre_step refines the
        # step-granularity resume point (current_round / current_step) so the
        # ledger no longer only sees round boundaries.
        self._ledger = ledger
        self._job = job

    def reasoning_session_id(self, step_id: str) -> str:
        """The stable, predictable kernel session id for a step.

        Stable across resume (same session + same step → same id), so the
        runtime can reference the reasoning session without copying it.
        """
        return f"{self._session_id}-{step_id}"

    def reasoning_storage_dir(self) -> str | None:
        """The kernel reasoning storage dir (``{store.base_dir}/reasoning``).

        ``None`` when the store does not expose a base directory (e.g. a
        non-file backend); the adapter then runs the session in-memory.
        """
        base = getattr(self._store, "base_dir", None)
        if base is None:
            return None
        return str(Path(base) / "reasoning")

    def on_pre_step(self, step: Any, context: Any) -> None:
        """Snapshot ``DomainState`` + step config + reasoning reference.

        Phase 4 adds the config half (``step_config``) alongside the state
        half (``domain_state``): a self-describing ``StepConfig`` snapshot so
        the step can later be exported / re-mounted (decision 27).
        """
        state = accumulate_state(context.step_results)
        reasoning_id = self.reasoning_session_id(step.id)
        step_type = self._step_types.get(step.step_type) if self._step_types else None
        self._store.save_snapshot(
            ResourceRef("step", step.id),
            {
                "step_id": step.id,
                "round_idx": self._round_idx,
                "reasoning_session_id": reasoning_id,
                "domain_state": state,
                "step_config": StepConfig.from_step(step, step_type).to_dict(),
            },
        )
        # Hand the stable reasoning-session id + storage dir to the executor
        # (adapter) so the kernel session is reused, not recreated.
        context.metadata[REASONING_SESSION_ID_KEY] = reasoning_id
        storage_dir = self.reasoning_storage_dir()
        if storage_dir is not None:
            context.metadata[REASONING_STORAGE_DIR_KEY] = storage_dir
        # R0a-2: refine the step-granularity resume point in the ledger.  The
        # recorded step is the *next* step to run — if the process dies here,
        # resume starts from it (at-least-once, FR-R4 idempotent).
        if self._ledger is not None and self._job is not None:
            self._ledger.update(
                self._job,
                self._session_id,
                current_round=self._round_idx,
                current_step=step.id,
            )
        return None  # keep the original step


class RoundCheckpoint:
    """Round-boundary checkpoints (called directly by the meta-loop)."""

    def __init__(self, store: IResourceStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    def round_start(
        self,
        round_idx: int,
        goal: str,
        input_state: dict[str, Any] | None,
        *,
        step_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self._store.save_snapshot(
            ResourceRef("round", str(round_idx), ("start",)),
            {
                "round_idx": round_idx,
                "goal": goal,
                "domain_state": input_state or {},
            },
        )
        # R0a-1: optionally persist the complete pipeline StepConfig list so a
        # resume can rebuild the whole pipeline without re-solving.  Static
        # pipelines pass it at round start; generator pipelines record it after
        # the planner step expands (deferred follow-up).
        if step_configs is not None:
            self.save_step_configs(round_idx, step_configs)

    def save_step_configs(
        self,
        round_idx: int,
        step_configs: list[dict[str, Any]],
    ) -> None:
        """Persist the round's full ``StepConfig`` list (R0a-1).

        Stored under ``round://{n}/step_configs`` (a separate sub-path so the
        start snapshot keeps its exact schema).
        """
        self._store.save_snapshot(
            ResourceRef("round", str(round_idx), ("step_configs",)),
            {
                "round_idx": round_idx,
                "step_configs": list(step_configs),
            },
        )

    def load_step_configs(self, round_idx: int) -> list[dict[str, Any]] | None:
        """Return the round's ``StepConfig`` list, or None when not recorded."""
        snapshot = self._store.load_snapshot(
            ResourceRef("round", str(round_idx), ("step_configs",))
        )
        if not snapshot:
            return None
        configs = snapshot.get("step_configs")
        return configs if isinstance(configs, list) else None

    def round_end(
        self,
        round_idx: int,
        goal_status: dict[str, Any] | None,
        output_state: dict[str, Any] | None,
    ) -> None:
        self._store.save_snapshot(
            ResourceRef("round", str(round_idx), ("end",)),
            {
                "round_idx": round_idx,
                "goal_status": goal_status or {},
                "domain_state": output_state or {},
            },
        )

    def load_start(self, round_idx: int) -> dict[str, Any] | None:
        return self._store.load_snapshot(ResourceRef("round", str(round_idx), ("start",)))

    def load_end(self, round_idx: int) -> dict[str, Any] | None:
        return self._store.load_snapshot(ResourceRef("round", str(round_idx), ("end",)))

