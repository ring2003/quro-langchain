"""RecoveryCoordinator — the R0 structural-resume owner (recovery-roadmap.md §2).

A thin ``match recovery_mode`` router, **not** a strategy framework.  R0
implements only ``recovery_mode="none"`` (structural re-run at the step
boundary, at-least-once).  ``compaction`` / ``mount`` are later roadmap phases
and raise ``NotImplementedError`` pointing at the roadmap.

The coordinator owns the recovery *trigger and strategy selection*; it does not
own execution — ``PipelineRunner`` / ``StepExecutor`` still execute.  Readiness
is inline fail-fast (each resume step names the missing input and where it
should live), with an optional read-only diagnostic — never a hard gate
(recovery-roadmap.md §2.3, runtime-session.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quro.core.domain.step_config import StepConfig
from quro.core.domain.step_factory import PipelineFactory
from quro.core.resources import (
    IResourceStore,
    JobKey,
    ResourceRef,
    RuntimeSessionLedger,
)
from quro.core.resources.offload import resolve_offloaded
from quro.core.resources.resume import resume_domain_state
from quro.pipeline.core import Pipeline, PipelineConfig, PipelineRunner
from quro.runtime.backtracker import Backtracker
from quro.runtime.inline_step import get_inline_step
from quro.steps.checkpoint_hook import RoundCheckpoint, restore_domain_state
from quro.steps.core import StepResult
from quro.steps.hooks import HookFactoryContext


class RecoveryResumeError(RuntimeError):
    """A resume input is missing or malformed (producer/consumer bisection)."""


@dataclass
class MissingData:
    """One absent resume input, with the location where it should live."""

    key: str
    where: str
    detail: str = ""


@dataclass
class ResumeResult:
    """The outcome of a single resume."""

    resumed_from_step: str
    skipped_steps: list[str] = field(default_factory=list)
    pipeline_result: Any = None


def rebuild_context_results(
    state: dict[str, Any],
    *,
    store: IResourceStore | None = None,
) -> dict[str, StepResult]:
    """Rebuild ``context_results`` from a restored ``DomainState`` (G4).

    A step is complete when it is marked ``completed`` in ``state["steps"]`` or
    owns an artifact (``state["artifacts"][].step_id``) — the union is used, so
    an artifact-carrying step is complete even if its status row is missing
    (weak/dangling-tolerant).  Each completed step yields one ``StepResult``
    whose ``state`` is the shared cumulative ``DomainState``; ``DependencyInjectHook``
    reads only ``artifacts``, so sharing the cumulative state is safe.
    Offloaded artifacts (a ``ref`` with no ``body``) are resolved through
    *store* when given.
    """
    completed: set[str] = set()
    for s in state.get("steps", []):
        if isinstance(s, dict) and s.get("status") == "completed" and s.get("step_id"):
            completed.add(str(s["step_id"]))

    by_step: dict[str, list[dict[str, Any]]] = {}
    for a in state.get("artifacts", []):
        if not isinstance(a, dict):
            continue
        sid = str(a.get("step_id") or "")
        if not sid:
            continue
        completed.add(sid)
        entry = dict(a)
        if "ref" in entry and "body" not in entry and store is not None:
            resolved = resolve_offloaded(store, entry)
            if isinstance(resolved, dict):
                entry = resolved
        by_step.setdefault(sid, []).append(entry)

    return {
        sid: StepResult.success(sid, state=state, artifacts=by_step.get(sid, []))
        for sid in completed
    }


def check_readiness(
    ledger: RuntimeSessionLedger,
    job: JobKey,
    session_id: str,
) -> list[MissingData]:
    """Return the missing resume inputs (read-only diagnostic, never a gate)."""
    missing: list[MissingData] = []
    data = ledger.load(job, session_id)
    if data is None:
        missing.append(
            MissingData(
                "ledger",
                f"jobs/{job.domain}/{job.problem_name}/{job.goal_uuid}/"
                f"sessions/{session_id}/index.json",
            )
        )
        return missing
    if not isinstance(data.get("current_round"), int):
        missing.append(MissingData("current_round", "ledger.current_round"))
    return missing


class RecoveryCoordinator:
    """Route a resume request and orchestrate the structural re-run (R0a-2)."""

    def __init__(
        self,
        *,
        store: IResourceStore,
        ledger: RuntimeSessionLedger,
        job: JobKey,
        session_id: str,
        step_types: Any = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._job = job
        self._session_id = session_id
        self._step_types = step_types

    def resume(
        self,
        *,
        recovery_mode: str = "none",
        step_executor: Any,
        problem: str | None = None,
        recovery_compaction: str | None = None,
        recovery_compaction_for_step_type: dict[str, str] | None = None,
    ) -> ResumeResult:
        """Read the resume point, restore state, rebuild, and hand off to the runner.

        Sequence (recovery-roadmap.md §3 R0a-2 / R1):
            1. read ledger ``current_round`` / ``current_step``;
            2. restore the application ``DomainState`` (step checkpoint first,
               round-start snapshot fallback, round events replayed);
            3. rebuild ``context_results`` from the restored state;
            4. rebuild the pipeline from the round-level ``step_configs``
               (no expander — the planner step is already in ``context_results``);
            5. for ``recovery_mode="compaction"``, inline the declared
               compaction act into the recovering step (primitive-step.md §4) —
               evaluate it against the backtracker's ``context_rebuild`` and
               substitute its instruction into the step's ``next_instruction``
               input;
            6. resume ``PipelineRunner`` from ``current_step`` (skip completed).
        """
        if recovery_mode not in ("none", "compaction"):
            raise NotImplementedError(
                f"recovery_mode={recovery_mode!r} is not implemented "
                "(R1 compaction) or R2 (mount). See recovery-roadmap.md."
            )
        if step_executor is None:
            raise RecoveryResumeError(
                "resume requires a step_executor (the executor still executes)"
            )

        data = self._ledger.load(self._job, self._session_id)
        if data is None:
            raise RecoveryResumeError(
                f"no session ledger at jobs/{self._job.domain}/"
                f"{self._job.problem_name}/{self._job.goal_uuid}/sessions/"
                f"{self._session_id}/index.json"
            )
        current_round = data.get("current_round")
        if not isinstance(current_round, int):
            raise RecoveryResumeError(
                "ledger has empty current_round — cannot locate the resume point"
            )
        current_step = str(data.get("current_step") or "")

        state = self._restore_state(current_round, current_step)
        context_results = rebuild_context_results(state, store=self._store)
        pipeline = self._rebuild_pipeline(current_round, step_executor)

        if recovery_mode == "compaction":
            self._inline_compaction(
                pipeline=pipeline,
                state=state,
                current_step=current_step,
                step_executor=step_executor,
                recovery_compaction=recovery_compaction,
                recovery_compaction_for_step_type=recovery_compaction_for_step_type or {},
            )

        problem = problem if problem is not None else self._goal(current_round)
        runner = PipelineRunner(
            pipeline,
            step_executor=step_executor,
            config=pipeline.config,
            backtracker=Backtracker(),
            step_types=self._step_types,
        )
        result = runner.run(problem, context_results=context_results)
        return ResumeResult(
            resumed_from_step=current_step or "(round start)",
            skipped_steps=sorted(context_results),
            pipeline_result=result,
        )

    # -- internals ----------------------------------------------------------

    def _inline_compaction(
        self,
        *,
        pipeline: Pipeline,
        state: dict[str, Any],
        current_step: str,
        step_executor: Any,
        recovery_compaction: str | None,
        recovery_compaction_for_step_type: dict[str, str],
    ) -> None:
        """Inline the declared compaction act into the recovering step
        (primitive-step.md §4).

        Evaluates the compaction act against the backtracker's
        ``context_rebuild`` and substitutes its instruction into the step's
        ``next_instruction`` input — a micro step-split inside the recovering
        step, no pipeline rebuild.
        """
        spec = pipeline.get_step(current_step)
        if spec is None:
            # Round-start resume, or current_step is not in the rebuilt pipeline
            # (e.g. already folded).  There is no recovering StepSpec to inject
            # into — plain structural resume.
            return
        step_type = spec.step_type

        name = recovery_compaction_for_step_type.get(step_type, recovery_compaction)
        if name is None:
            # No compaction declared → plain structural resume (current behaviour).
            return

        inline_step = get_inline_step(name)
        if inline_step is None:
            from quro.runtime.inline_step import default_registry

            raise RecoveryResumeError(
                f"unknown recovery compaction {name!r} — registered inline steps: "
                f"{', '.join(default_registry().names()) or '(none)'}"
            )

        backend = getattr(step_executor, "backend", None)
        if backend is None:
            raise RecoveryResumeError(
                f"recovery_mode='compaction' needs an LLM backend on the "
                f"step_executor (compaction '{name}' is a one-shot call with no "
                "kernel session); the executor exposes none"
            )
        if not hasattr(backend, "complete"):
            raise RecoveryResumeError(
                f"step_executor backend for compaction '{name}' does not "
                "implement complete()"
            )

        report = Backtracker().rebuild(state, failed_step_id=current_step)

        instruction = inline_step.eval(
            context=report.context_rebuild,
            backend=backend,
            step_type=step_type,
            objective=spec.objective,
            step_id=current_step,
        )

        spec.next_instruction = instruction

    def _restore_state(self, current_round: int, current_step: str) -> dict[str, Any]:
        # Prefer the step-pre checkpoint of current_step (state-complete).
        if current_step:
            state = restore_domain_state(self._store, current_step)
            if state is not None:
                return state
        # Fall back to the round-start snapshot, replaying round events on top.
        return resume_domain_state(
            self._store,
            ResourceRef("round", str(current_round), ("start",)),
            event_ref=ResourceRef("round", str(current_round)),
        )

    def _rebuild_pipeline(self, current_round: int, step_executor: Any) -> Pipeline:
        checkpoint = RoundCheckpoint(self._store, self._session_id)
        raw = checkpoint.load_step_configs(current_round)
        if not raw:
            raise RecoveryResumeError(
                f"no step_configs at round://{current_round}/step_configs — "
                "cannot rebuild the pipeline without re-solve (R0a-1 record missing)"
            )
        configs = [StepConfig.from_dict(c) for c in raw]
        rebuilt = PipelineFactory().rebuild(
            configs,
            hook_context=HookFactoryContext(step_executor=step_executor),
        )
        return Pipeline(
            name=f"round-{current_round}",
            steps=[r.spec for r in rebuilt],
            config=PipelineConfig(),
            expander=None,  # G5: never re-expand on resume
        )

    def _goal(self, current_round: int) -> str:
        snapshot = self._store.load_snapshot(
            ResourceRef("round", str(current_round), ("start",))
        )
        if snapshot and isinstance(snapshot.get("goal"), str):
            return snapshot["goal"]
        return ""
