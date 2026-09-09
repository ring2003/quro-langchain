"""StepFactory / PipelineFactory — the rebuild direction (phase 5).

A factory turns a snapshot :class:`StepConfig` back into live objects
(:class:`~quro.steps.core.StepSpec` + :class:`StepType`) so an exported /
mounted step can be re-run.  It is the *input* to mount execution, cross-model
run, and fractal mount — all of which stay out of scope here (they are
consumers, per ``unified-resource-layer-phase5-implementation-plan.md`` §1).

Rules (locked in the phase-5 roadmap):

- ``StepConfig`` is the only input (decision 31) — the factory never assembles
  a ``StepSpec`` from a domain catalog.
- ``step_type`` is rebuilt *anonymously* from the config's inlined fields
  (``skill_pool`` / ``identity_block`` / ``executor_hints`` / ``features``).
- Explicit hooks (``pre_hooks`` / ``post_hooks`` names) are resolved through
  ``HookRegistry.create(name, context)``; context-dependent hooks such as
  ``RecoveryHook`` are *not* in the config — the pipeline re-attaches them from
  ``Feature.RECOVERY`` (decision 34).
- Rebuild is structural only: it never schedules or executes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quro.core.domain.step_config import StepConfig
from quro.core.domain.step_type import StepType
from quro.steps.core import StepSpec
from quro.steps.hooks import HookFactoryContext, HookRegistry, get_hook_registry


@dataclass
class RebuiltStep:
    """What mount execution needs: a ``StepSpec`` + its rebuilt ``StepType``."""

    spec: StepSpec
    step_type: StepType


@runtime_checkable
class IStepFactory(Protocol):
    """Rebuild one ``StepConfig`` into a ``RebuiltStep``."""

    def rebuild(
        self,
        config: StepConfig,
        *,
        hook_context: HookFactoryContext,
    ) -> RebuiltStep: ...


class StepFactory:
    """Rebuild ``StepConfig`` into ``StepSpec`` + anonymous ``StepType``.

    Hooks are resolved by name through a ``HookRegistry`` (the global one by
    default).  ``rebuild`` is an execution-front assembly point: the caller
    supplies a populated ``HookFactoryContext`` (phase-5 §4.3).
    """

    def __init__(self, registry: HookRegistry | None = None) -> None:
        self._registry = registry if registry is not None else get_hook_registry()

    def rebuild(
        self,
        config: StepConfig,
        *,
        hook_context: HookFactoryContext,
    ) -> RebuiltStep:
        return RebuiltStep(
            spec=self._rebuild_spec(config, hook_context),
            step_type=self._rebuild_step_type(config),
        )

    def _rebuild_spec(
        self,
        config: StepConfig,
        hook_context: HookFactoryContext,
    ) -> StepSpec:
        from quro.core.tools.grant import resolve_capabilities

        return StepSpec(
            id=config.id,
            name=config.name,
            objective=config.objective,
            tools=list(resolve_capabilities(config.tools)),
            depends_on=list(config.depends_on),
            expected_output=config.expected_output,
            validation=config.validation,
            timeout=config.timeout,
            inputs=list(config.inputs),
            checklist=[dict(c) for c in config.checklist],
            sub_steps=[self._rebuild_spec(s, hook_context) for s in config.sub_steps],
            pre_hooks=[
                self._registry.create(name, hook_context) for name in config.pre_hooks
            ],
            post_hooks=[
                self._registry.create(name, hook_context) for name in config.post_hooks
            ],
            message_strategy=config.message_strategy,
            hil=config.hil,
            step_type=config.step_type,
            skills=list(config.skills),
            grant_name=config.grant_name,
            exclude=resolve_capabilities(config.exclude),
            features=tuple(config.features),
            mounted=config.mounted,
        )

    @staticmethod
    def _rebuild_step_type(config: StepConfig) -> StepType:
        from quro.core.tools.grant import resolve_capabilities

        return StepType(
            name=config.step_type,
            skill_pool=list(config.skill_pool),
            identity_block=config.identity_block,
            executor_hints=dict(config.executor_hints),
            features=tuple(config.features),
            tools=resolve_capabilities(config.tools),
            grant_name=config.grant_name,
            exclude=resolve_capabilities(config.exclude),
            mounted=config.mounted,
        )


@runtime_checkable
class IPipelineFactory(Protocol):
    """Rebuild a list of ``StepConfig`` into a topologically-ordered step list."""

    def rebuild(
        self,
        configs: list[StepConfig],
        *,
        hook_context: HookFactoryContext,
    ) -> list[RebuiltStep]: ...


class PipelineFactory:
    """Rebuild a ``StepConfig`` list into topologically-ordered ``RebuiltStep``s.

    ``depends_on`` edges come from ``config.depends_on`` (name-only).  It does
    not schedule or execute — that stays with the runtime (phase-5 §4.2).
    """

    def __init__(self, registry: HookRegistry | None = None) -> None:
        self._step_factory = StepFactory(registry)

    def rebuild(
        self,
        configs: list[StepConfig],
        *,
        hook_context: HookFactoryContext,
    ) -> list[RebuiltStep]:
        ordered = _topological_order(configs)
        return [
            self._step_factory.rebuild(c, hook_context=hook_context) for c in ordered
        ]


def _topological_order(configs: list[StepConfig]) -> list[StepConfig]:
    """Return *configs* in dependency order (Kahn's algorithm).

    Raises:
        ValueError: On a duplicate id, a reference to an unknown step, or a
            dependency cycle.
    """
    by_id: dict[str, StepConfig] = {}
    for c in configs:
        if c.id in by_id:
            raise ValueError(f"duplicate step id '{c.id}'")
        by_id[c.id] = c

    indegree: dict[str, int] = {c.id: 0 for c in configs}
    dependents: dict[str, list[str]] = {c.id: [] for c in configs}
    for c in configs:
        for dep in c.depends_on:
            if dep not in by_id:
                raise ValueError(f"step '{c.id}' depends on unknown step '{dep}'")
            indegree[c.id] += 1
            dependents[dep].append(c.id)

    ready: deque[str] = deque(c.id for c in configs if indegree[c.id] == 0)
    ordered: list[StepConfig] = []
    while ready:
        sid = ready.popleft()
        ordered.append(by_id[sid])
        for dependent in dependents[sid]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(ordered) != len(configs):
        remaining = sorted(set(by_id) - {c.id for c in ordered})
        raise ValueError(f"cycle detected in step dependencies involving: {remaining}")
    return ordered
