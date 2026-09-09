"""Pipeline definition and orchestration.

Introduces a declarative ``Pipeline`` (a DAG of ``StepSpec``s bound to step
types) and a ``PipelineRunner`` that executes the steps in dependency order
with log prefixes, model-switch confirmation, and optional dynamic replanning
hooks.
"""

from __future__ import annotations

import enum
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from quro.core.confirm import confirm_model_switch
from quro.core.domain.mount import check_mount
from quro.core.domain.step_factory import PipelineFactory
from quro.core.features import Feature
from quro.steps.core import StepExecutor, StepResult, StepSpec
from quro.steps.hooks import HookChain, HookContext, HookFactoryContext, get_hook_registry
from quro.steps.hil_step import HILBackend, HumanInLoopStep

logger = logging.getLogger(__name__)


class PipelineValidationError(ValueError):
    """Raised when a pipeline fails structural validation (cycles, refs)."""


# ---------------------------------------------------------------------------
# PlannerAction / PlannerHook — dynamic replanning protocol
# ---------------------------------------------------------------------------


class PlannerAction(enum.Enum):
    """Action returned by a ``PlannerHook`` after a step completes."""

    CONTINUE = "continue"  # proceed to the next step in the pipeline
    REPLAN = "replan"      # stop the pipeline and return to the planner
    ABORT = "abort"        # stop the pipeline with a terminal failure


class PlannerHook(Protocol):
    """Hook called after each step finishes so a policy loop can intervene.

    Implementations inspect the step result and accumulated context and
    return a ``PlannerAction`` to continue, replan, or abort.
    """

    def on_step_complete(
        self,
        step: StepSpec,
        result: StepResult,
        context: dict[str, Any],
    ) -> PlannerAction:
        """Inspect *result* and return the next action.

        Args:
            step: The step that just completed.
            result: The ``StepResult`` produced by *step*.
            context: The pipeline context dict including ``problem`` and
                ``results`` of all executed steps so far.
        """
        ...


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Runtime configuration shared by all steps of a pipeline.

    Args:
        default_model: Model used by the backend unless a step type
            declares its own model.
        max_rounds: Maximum reasoning rounds per step execution.
        context_budget_chars: Character budget for the projected context.
        state_dir: Directory for session/artifact persistence.
        stop_on_failure: Stop the pipeline as soon as a step fails. If False,
            remaining steps still run (dependencies may then fail).
        auto_confirm: Skip model-switch confirmation prompts (for CI/non-TTY).
    """

    default_model: str = "gpt-4o-mini"
    max_rounds: int = 12
    context_budget_chars: int = 12000
    state_dir: str = ".quro/pipeline"
    stop_on_failure: bool = True
    auto_confirm: bool = False
    default_message_strategy: str = "rebuild"
    """Default message strategy for all steps in this pipeline.
    ``"rebuild"`` or ``"accumulate"``."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    """A declarative collection of steps executed in dependency order.

    Args:
        name: Pipeline name (used in logs/results).
        description: Human-readable description.
        steps: Ordered list of ``StepSpec`` (declared order is the default
            order for steps without dependencies).
        default_model: Model used by the backend unless overridden.
        config: Pipeline-level runtime configuration.
    """

    name: str = ""
    description: str = ""
    steps: list[StepSpec] = field(default_factory=list)
    default_model: str = "gpt-4o-mini"
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for s in self.steps}

    def get_step(self, step_id: str) -> StepSpec | None:
        """Return the step with *step_id*, or None."""
        return self._by_id.get(step_id)

    def _namespace_step(
        self,
        step: StepSpec,
        prefix: str,
        seen: list[StepSpec],
    ) -> None:
        """Rewrite ``step`` (and its ``sub_steps``) under ``prefix`` in place.

        Ids become ``prefix/inner`` and ``depends_on`` references are rewritten
        to their namespaced form, so a flattened (mounted) unit's steps cannot
        collide with the outer pipeline or with a sibling mount of the same
        frozen pipeline.  ``seen`` accumulates the top-level rewritten steps
        (for mount flattening; recursion appends nested specs to the parent).
        """
        step.id = f"{prefix}/{step.id}"
        step.depends_on = [
            f"{prefix}/{d}" if "/" not in d else d for d in step.depends_on
        ]
        for sub in step.sub_steps:
            self._namespace_step(sub, prefix, seen)
        seen.append(step)

    def validate(self) -> list[str]:
        """Validate the pipeline structure.

        Checks duplicate step ids, unknown ``depends_on`` references, and
        dependency cycles.

        Returns:
            A list of error messages; empty means the pipeline is valid.
        """
        errors: list[str] = []

        if len(self._by_id) != len(self.steps):
            ids = [s.id for s in self.steps]
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            errors.append(f"duplicate step ids: {dupes}")

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in self._by_id:
                    errors.append(f"step '{step.id}' depends on unknown step '{dep}'")

        try:
            self.topological_order()
        except PipelineValidationError as e:
            errors.append(str(e))

        return errors

    def topological_order(self) -> list[StepSpec]:
        """Return steps in dependency order (Kahn's algorithm).

        Raises:
            PipelineValidationError: If the dependency graph contains a cycle,
                a duplicate id, or a reference to an unknown step.
        """
        indegree: dict[str, int] = {s.id: 0 for s in self.steps}
        dependents: dict[str, list[str]] = {s.id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in self._by_id:
                    raise PipelineValidationError(
                        f"step '{step.id}' depends on unknown step '{dep}'"
                    )
                indegree[step.id] += 1
                dependents[dep].append(step.id)

        ready: deque[str] = deque(s.id for s in self.steps if indegree[s.id] == 0)
        ordered: list[str] = []

        while ready:
            step_id = ready.popleft()
            ordered.append(step_id)
            for dependent in dependents[step_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(ordered) != len(self.steps):
            remaining = sorted(set(self._by_id) - set(ordered))
            raise PipelineValidationError(
                f"cycle detected in step dependencies involving: {remaining}"
            )

        return [self._by_id[sid] for sid in ordered]


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Result of running a ``Pipeline``.

    Args:
        ok: True if all executed steps succeeded (no stop_on_failure abort).
        step_results: Mapping of step id to its ``StepResult``.
        order: The execution order of step ids.
        error: The first failure message, if any.
        metrics: Aggregated metrics across steps.
        partial: True when the pipeline was interrupted for replanning.
        replan_reason: If partial=True, the reason for replanning.
    """

    ok: bool
    step_results: dict[str, StepResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    partial: bool = False
    replan_reason: str | None = None
    backtrack: dict[str, Any] | None = None
    """Backtracker report (Phase C §2.7) when the pipeline terminated
    abnormally — a structural round-graph rebuild for the planner."""

    @classmethod
    def invalid(cls, errors: list[str]) -> PipelineResult:
        """A result for a pipeline that failed structural validation."""
        return cls(
            ok=False,
            error="; ".join(errors),
            metrics={"validation_errors": errors},
        )

    @classmethod
    def partial_result(
        cls,
        step_results: dict[str, StepResult],
        order: list[str],
        *,
        replan_reason: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """A result for a pipeline that was interrupted for replanning.

        The pipeline completed some steps successfully but the planner
        should re-evaluate the remaining plan.
        """
        return cls(
            ok=True,
            step_results=step_results,
            order=order,
            partial=True,
            replan_reason=replan_reason,
            metrics=metrics or {},
        )

    def succeeded_steps(self) -> list[str]:
        """Ids of steps that completed successfully."""
        return [sid for sid, r in self.step_results.items() if r.ok]


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------


class PipelineRunner:
    """Executes a ``Pipeline`` step-by-step in dependency order.

    Args:
        pipeline: The pipeline to run.
        step_executor: The ``StepExecutor`` that runs each step (required).
            Typically a ``StepAdapter`` instance.
        config: Optional per-run ``PipelineConfig`` overriding the pipeline's.
        planner_hook: Optional ``PlannerHook`` for dynamic replanning.
        hil_backend: Optional ``HILBackend`` for human-in-the-loop steps.
        step_types: Optional StepTypeCatalog for runtime resolution of features
            from StepType when ``step.features`` is empty (create_step fix).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        step_executor: StepExecutor,
        config: PipelineConfig | None = None,
        planner_hook: PlannerHook | None = None,
        hil_backend: HILBackend | None = None,
        backtracker: Any = None,
        checkpoint_hook_factory: Callable[[], Any] | None = None,
        mount_resolver: Any = None,
        step_types: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or pipeline.config
        self.planner_hook = planner_hook
        self.hil_backend = hil_backend
        self.step_executor = step_executor
        self.backtracker = backtracker
        self.checkpoint_hook_factory = checkpoint_hook_factory
        # reuse-mount (mount-semantics.md §5): resolves a step's ``MountRef``
        # into the foreign frozen ``[StepConfig]`` to flatten. ``None`` = no
        # mounted steps (ordinary pipelines).
        self.mount_resolver = mount_resolver
        self.step_types = step_types

    def run(
        self,
        problem: str,
        *,
        context_results: dict[str, StepResult] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline for *problem*.

        Flow:
        1. Validate the pipeline (DAG cycle check, dependency references).
        2. For each step in topological order:

           a. Build HookContext and run pre-hooks (step may be replaced).
           b. If the step is a Human-in-Loop step (``hil == True``),
              pause the pipeline and wait for user input.
           c. Execute the (possibly mutated) step via the step executor.
           d. Run post-hooks (result may be replaced).

        4. Collect artifacts and results; honor ``stop_on_failure``.
        5. If ``planner_hook`` is set, call it after each step and
           return a partial result on ``REPLAN``.

        Args:
            problem: The problem statement given to the pipeline.
            context_results: Pre-existing step results from a prior
                partial execution. Steps already in this dict are
                skipped, so the pipeline resumes from where it left off.

        Returns:
            A ``PipelineResult`` aggregating per-step outcomes.
        """
        errors = self.pipeline.validate()
        if errors:
            return PipelineResult.invalid(errors)

        try:
            ordered = self.pipeline.topological_order()
        except PipelineValidationError as e:
            return PipelineResult.invalid([str(e)])

        # Attach the post-step recovery hook to recovery-enabled step types
        # (both static steps and, later, dynamically expanded executor steps),
        # and the pre-step checkpoint hook via the assembler's factory (G6).
        for s in ordered:
            self._attach_recovery_hook(s)
            self._attach_checkpoint_hook(s)

        pipeline_name = self.pipeline.name or "pipeline"
        total_steps = len(ordered)
        print(f"\n[{pipeline_name}] Starting — {total_steps} step(s) to execute")
        if context_results:
            skipped = sum(1 for s in ordered if s.id in context_results)
            if skipped:
                print(f"[{pipeline_name}] Resuming — {skipped} step(s) already complete")

        results: dict[str, StepResult] = dict(context_results or {})
        context: dict[str, Any] = {"problem": problem, "results": results}
        error: str | None = None
        stopped_early = False

        pipeline_name = self.pipeline.name or "pipeline"
        prev_step_label: str | None = None
        prev_model: str | None = None

        # Step placeholder for early-exit reference
        step: StepSpec | None = None  # noqa: F811

        # Pending queue.  It grows at run time when the planner step expands
        # into executor steps (dynamic step insertion, see below).
        pending = deque(s for s in ordered if s.id not in results)

        while pending:
            step = pending.popleft()
            if step.id in results:
                continue

            # ---------------------------------------------------------------
            # Build hook context and run pre-hooks
            # ---------------------------------------------------------------
            hook_ctx = HookContext(problem=problem, step_results=results)
            chain = HookChain(step.pre_hooks + step.post_hooks)
            step = chain.run_pre_hooks(step, hook_ctx)

            # Apply hook context overrides to the step executor
            context["_hook_context"] = hook_ctx

            # ---------------------------------------------------------------
            # reuse-mount (mount-semantics.md §5): a step whose body is a
            # foreign frozen pipeline expands in place — resolve, check,
            # rebuild, flatten, and fold the inner product back under the
            # outer step_id (closure law §8.2).  No separate subagent, no
            # second LLM instance — the inner steps run through the *same*
            # step_executor.
            # ---------------------------------------------------------------
            if step.mounted is not None and self.mount_resolver is not None:
                result = self._execute_mounted_step(step, context, hook_ctx, results)
                results[step.id] = result
                status = "OK" if result.ok else "FAIL"
                print(f"[{pipeline_name}] Step {step.id} (mounted) — {status}")
                if not result.ok and self.config.stop_on_failure:
                    error = result.error or f"step '{step.id}' failed"
                    break
                continue

            # ---------------------------------------------------------------
            # Handle human-in-the-loop step
            # ---------------------------------------------------------------
            if step.hil:
                print(f"\n[{pipeline_name}] Step {step.id} (human) — Waiting for input")
                result = self._execute_hil_step(step, context)
                results[step.id] = result
                if not result.ok and self.config.stop_on_failure:
                    error = result.error
                    break
                continue

            # ---------------------------------------------------------------
            # Model switch confirmation
            # ---------------------------------------------------------------
            step_label = step.step_type or step.id
            step_model = self.config.default_model
            task_id = f"{pipeline_name}/{step.id}"

            if prev_step_label and prev_step_label != step_label and prev_model != step_model:
                if not confirm_model_switch(
                    task_id=task_id,
                    from_model=prev_model,
                    to_model=step_model,
                    auto_confirm=self.config.auto_confirm,
                ):
                    error = f"User aborted at step '{step.id}'"
                    break

            # ---------------------------------------------------------------
            # Log prefix
            # ---------------------------------------------------------------
            model_info = f"model={step_model}" if step_model != prev_model else ""
            print(f"\n[{pipeline_name}] Step {step.id} ({step_label}) {model_info}".rstrip())
            # Show what the step worker actually sees — the self-contained task.
            objective_preview = step.objective[:200].replace("\n", " ")
            print(f"  Task: {objective_preview}")
            sys.stdout.flush()

            prev_step_label = step_label
            prev_model = step_model

            # ---------------------------------------------------------------
            # Execute step
            # ---------------------------------------------------------------
            try:
                result = self.step_executor.execute(step, context)
            except Exception as e:  # noqa: BLE001 - step failure must not kill pipeline
                logger.exception("Step '%s' raised during execution", step.id)
                result = StepResult.failure(step.id, f"{type(e).__name__}: {e}")
            results[step.id] = result

            status = "OK" if result.ok else "FAIL"
            print(f"[{pipeline_name}] Step {step.id} — {status}")

            # ---------------------------------------------------------------
            # Run post-hooks
            # ---------------------------------------------------------------
            result = chain.run_post_hooks(step, result, hook_ctx)

            # ---------------------------------------------------------------
            # Planner hook check
            # ---------------------------------------------------------------
            if self.planner_hook is not None:
                action = self.planner_hook.on_step_complete(step, result, context)
                if action == PlannerAction.REPLAN:
                    stopped_early = True
                    break
                if action == PlannerAction.ABORT:
                    error = result.error or f"step '{step.id}' aborted by planner"
                    stopped_early = True
                    break

            if not result.ok and self.config.stop_on_failure:
                error = result.error or f"step '{step.id}' failed"
                logger.warning("Pipeline stopped after failed step '%s': %s", step.id, error)
                break

        # Final order.
        try:
            final_ordered = self.pipeline.topological_order()
        except PipelineValidationError:
            final_ordered = list(self.pipeline.steps)

        if stopped_early and self.planner_hook is not None:
            executed_order = [s.id for s in final_ordered if s.id in results]
            reason = ""
            if not error and step is not None:
                last_result = results.get(step.id)
                if last_result and last_result.replan_request:
                    reason = last_result.replan_request.get("reason", "")
            return PipelineResult.partial_result(
                results,
                executed_order,
                replan_reason=reason or "replan requested",
                metrics=self._aggregate_metrics(final_ordered, results),
            )

        ok = all(r.ok for r in results.values())
        metrics = self._aggregate_metrics(final_ordered, results)

        # Phase C (§2.7): on abnormal termination, wake the backtracker and
        # attach a structural round-graph rebuild for the planner.
        backtrack: dict[str, Any] | None = None
        if error and self.backtracker is not None:
            failed_step_id = step.id if step is not None else ""
            backtrack = self._backtrack_report(results, failed_step_id)

        return PipelineResult(
            ok=ok,
            step_results=results,
            order=[s.id for s in final_ordered],
            error=error,
            metrics=metrics,
            backtrack=backtrack,
        )

    def _attach_recovery_hook(self, step: StepSpec) -> None:
        """Attach a ``RecoveryHook`` to *step* when it is recovery-enabled.

        Idempotent: skips steps already carrying a ``recovery`` post-hook.
        This is the single wiring point for the post-step recovery hook
        (blueprint §6 / decision 12) — recovery enablement is declared per
        step type via ``StepType.features`` and mirrored onto the ``StepSpec``
        (phase 4).

        Phase 5 (decision 33/34): the hook is constructed through the
        ``HookRegistry`` from its name, passing the pipeline's step executor via
        a ``HookFactoryContext``.  Attachment stays feature-driven — the hook is
        never written into ``StepConfig``.
        """
        # Resolve features from StepType when step.features is empty
        # (create_step fix: runtime resolution of StepType properties).
        features = step.features
        if not features and self.step_types and getattr(step, "step_type", ""):
            features = self.step_types.features_for(step.step_type)

        if Feature.RECOVERY.value not in features:
            return
        if any(getattr(h, "name", "") == "recovery" for h in step.post_hooks):
            return
        from quro.steps.recovery_hook import recovery_hook_factory

        registry = get_hook_registry()
        if "recovery" not in registry.list_names():
            registry.register("recovery", recovery_hook_factory)
        ctx = HookFactoryContext(step_executor=self.step_executor)
        step.post_hooks.append(registry.create("recovery", ctx))

    def _attach_checkpoint_hook(self, step: StepSpec) -> None:
        """Attach the assembler's ``CheckpointHook`` as the first pre-hook (G6).

        Idempotent: skips steps already carrying a ``checkpoint`` pre-hook (so
        an assembler that pre-attached hooks, e.g. demo11, is not double-wired).
        The hook is **context-dependent** (store / session_id / round_idx), so
        it is supplied as a factory — never rebuilt by name through
        ``HookRegistry`` (a bare name would raise ``KeyError`` in
        ``PipelineFactory.rebuild``; recovery-r0-implementation-plan.md §2.5).
        """
        if self.checkpoint_hook_factory is None:
            return
        if any(getattr(h, "name", "") == "checkpoint" for h in step.pre_hooks):
            return
        step.pre_hooks.insert(0, self.checkpoint_hook_factory())

    def _execute_mounted_step(
        self,
        step: StepSpec,
        context: dict[str, Any],
        hook_ctx: HookContext,
        results: dict[str, StepResult],
    ) -> StepResult:
        """Flatten a mounted step's foreign pipeline and fold the product back.

        The §5 execution chain: resolve the ``MountRef`` → ``check_mount``
        (declared type check) → ``PipelineFactory.rebuild`` → run the inner
        steps through the *same* ``step_executor`` (flatten) → fold the inner
        product under the **outer** ``step_id`` (closure law).  Inner steps are
        namespace-prefixed (`outer/inner`) so the frozen unit's ids cannot
        collide with the outer pipeline; the folded result is keyed under the
        outer id so the outer's dependents read it exactly as if it were an
        ordinary step.
        """
        ref = step.mounted
        try:
            configs = self.mount_resolver.resolve(ref)
        except KeyError as e:
            return StepResult.failure(step.id, f"mount resolve failed: {e}")

        errors = check_mount(configs, step)
        if errors:
            return StepResult.failure(
                step.id, "mount check failed: " + "; ".join(errors)
            )

        rebuilt = PipelineFactory().rebuild(
            configs,
            hook_context=HookFactoryContext(step_executor=self.step_executor),
        )
        inner_specs = [r.spec for r in rebuilt]

        # Namespace the inner ids so the frozen unit's ids cannot collide with
        # the outer pipeline (or with a sibling mount of the same pipeline).
        # Inner ``depends_on`` edges are rewritten to the namespaced ids so
        # dependency injection still resolves between inner steps.
        namespaced: list[StepSpec] = []
        for inner in inner_specs:
            self.pipeline._namespace_step(inner, step.id, namespaced)
        inner_specs = namespaced

        # Flatten: run the inner sequence through the same executor.  A shared
        # ``results`` view lets an inner step read the outer context and its
        # inner predecessors (dependency injection), exactly like re-enqueued
        # steps would — but without re-toposorting the outer DAG.
        inner_results: dict[str, StepResult] = {}
        merged_results: dict[str, StepResult] = dict(results)
        for inner in inner_specs:
            ctx = HookContext(
                problem=context.get("problem", ""),
                step_results=merged_results,
            )
            chain = HookChain(inner.pre_hooks + inner.post_hooks)
            inner = chain.run_pre_hooks(inner, ctx)
            try:
                res = self.step_executor.execute(inner, context)
            except Exception as e:  # noqa: BLE001
                logger.exception("mounted step '%s' raised", inner.id)
                res = StepResult.failure(inner.id, f"{type(e).__name__}: {e}")
            res = chain.run_post_hooks(inner, res, ctx)
            folded = StepResult(
                step_id=inner.id,
                ok=res.ok,
                state=res.state,
                artifacts=res.artifacts,
                error=res.error,
                metrics=res.metrics,
            )
            inner_results[inner.id] = folded
            merged_results[inner.id] = folded

        ok = all(r.ok for r in inner_results.values())
        first_error = next(
            (r.error for r in inner_results.values() if not r.ok), None
        )

        # Fold (§5 step 5): the inner product becomes the outer artifact —
        # artifacts are re-owned to the outer ``step_id``, and the cumulative
        # state carries over so dependents see a complete step.
        artifacts: list[dict[str, Any]] = []
        state = None
        for r in inner_results.values():
            if r.state is not None:
                state = r.state
            for a in r.artifacts:
                owned = dict(a)
                owned["step_id"] = step.id
                artifacts.append(owned)

        return StepResult(
            step_id=step.id,
            ok=ok,
            state=state,
            artifacts=artifacts,
            error=first_error,
        )

    def _backtrack_report(
        self,
        results: dict[str, StepResult],
        failed_step_id: str,
    ) -> dict[str, Any] | None:
        """Build a round-object from the pipeline and ask the backtracker."""
        steps: list[dict[str, Any]] = []
        for spec in self.pipeline.steps:
            result = results.get(spec.id)
            if result is not None and result.ok:
                status = "completed"
            elif spec.id == failed_step_id:
                status = "in_progress"
            else:
                status = "pending"
            steps.append({
                "step_id": spec.id,
                "objective": spec.objective,
                "step_type": spec.step_type,
                "status": status,
                "depends_on": list(spec.depends_on),
                "access": None,
            })

        artifacts: list[dict[str, Any]] = []
        for result in results.values():
            for artifact in (result.artifacts or []):
                if isinstance(artifact, dict):
                    artifacts.append(artifact)

        # Carry the step-scoped recovery journals (blueprint §6) so the
        # backtracker can surface clue chains for rebuilding lost artifacts.
        recovery: dict[str, Any] = {}
        for result in results.values():
            rs = result.state or {}
            for step_id, journal in (rs.get("recovery") or {}).items():
                if isinstance(journal, dict):
                    recovery.setdefault(str(step_id), journal)

        state: dict[str, Any] = {
            "steps": steps,
            "artifacts": artifacts,
            "executing_step_id": failed_step_id,
            "phase": "step_execute",
            "recovery": recovery,
        }

        try:
            report = self.backtracker.rebuild(state, failed_step_id=failed_step_id)
            return report.to_dict() if hasattr(report, "to_dict") else dict(report)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("backtracker failed: %s", exc)
            return None

    def _aggregate_metrics(
        self,
        ordered: list[StepSpec],
        results: dict[str, StepResult],
    ) -> dict[str, Any]:
        executed = [s for s in ordered if s.id in results]
        return {
            "total_steps": len(ordered),
            "executed": len(executed),
            "succeeded": sum(1 for r in results.values() if r.ok),
            "failed": sum(1 for r in results.values() if not r.ok),
            "skipped": len(ordered) - len(executed),
        }

    def _execute_hil_step(
        self, step: StepSpec, context: dict[str, Any]
    ) -> StepResult:
        """Execute a human-in-the-loop step: pause and wait for user input.

        Args:
            step: The HIL step spec (``hil == True``).
            context: The pipeline context dict.

        Returns:
            A ``StepResult`` with the user's response as an artifact.
        """
        if self.hil_backend is None:
            return StepResult.failure(
                step.id,
                "No HIL backend configured — cannot execute human step",
            )

        hil_step = (
            step if isinstance(step, HumanInLoopStep)
            else HumanInLoopStep(
                id=step.id,
                name=step.name,
                objective=step.objective,
                depends_on=step.depends_on,
                choices=getattr(step, "choices", []),
                timeout=getattr(step, "timeout", 0),
            )
        )

        question = hil_step.render_prompt(context)

        try:
            response = self.hil_backend.ask_user(
                question=question,
                choices=hil_step.choices or None,
                timeout=hil_step.timeout,
            )
        except Exception as exc:
            logger.exception("HIL step '%s' failed", step.id)
            return StepResult.failure(step.id, f"HIL backend error: {exc}")

        artifact = {
            "artifact_id": f"hil_{step.id}",
            "kind": "human_input",
            "summary": f"Human response for '{step.id}'",
            "body": response,
        }
        return StepResult.success(
            step.id,
            state={"human_response": response},
            artifacts=[artifact],
        )
