"""Step hook protocol, context, chain, and registry.

Pre/post hooks allow intercepting and modifying step execution.  Hooks
are composable — a single step can have multiple pre-hooks and post-hooks,
executed in registration order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from quro.steps.core import StepResult, StepSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HookContext — mutable context passed through the hook chain
# ---------------------------------------------------------------------------


@dataclass
class HookContext:
    """Mutable context passed through the hook chain.

    Hooks read and write this context to influence step execution.
    The pipeline runner reads the final state after all pre-hooks have run
    and uses it to configure the executor.
    """

    problem: str
    """The top-level problem statement."""

    step_results: dict[str, Any] = field(default_factory=dict)
    """Results of all previously executed steps (StepResult keyed by step id)."""

    system_prompt: str | None = None
    """Override the system prompt for this step."""

    user_prompt: str | None = None
    """Override the user prompt for this step."""

    extra_tools: list[Any] = field(default_factory=list)
    """Additional tools to inject for this step."""

    remove_tools: set[str] = field(default_factory=set)
    """Tool names to exclude for this step."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary hook-to-hook / hook-to-executor passthrough data."""

    hil_pending: bool = False
    """When True, the pipeline runner pauses and waits for human input."""

    hil_question: str = ""
    """The question to show the user."""

    hil_response: str = ""
    """The user's response, filled by the pipeline runner after pause."""

    hil_options: list[str] = field(default_factory=list)
    """Optional constrained choices for the user."""

    @classmethod
    def from_pipeline_context(
        cls, problem: str, results: dict[str, Any] | None = None
    ) -> HookContext:
        """Create a HookContext from the standard pipeline context dict."""
        return cls(problem=problem, step_results=dict(results or {}))


# ---------------------------------------------------------------------------
# StepHook protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StepHook(Protocol):
    """Pre/post hook for step execution.

    A hook may implement either or both methods.  The pipeline runner calls
    on_pre_step before execution and on_post_step after execution.
    """

    name: str
    """Human-readable hook name for logging and debugging."""

    def on_pre_step(
        self,
        step: StepSpec,
        context: HookContext,
    ) -> StepSpec | None:
        """Called before the step executes.

        Return a *new* StepSpec to replace the step being executed, or None
        to keep the original.  The HookContext is mutable and can be used to
        inject the system prompt, add/remove tools, etc.
        """
        ...

    def on_post_step(
        self,
        step: StepSpec,
        result: StepResult,
        context: HookContext,
    ) -> StepResult | None:
        """Called after the step executes.

        Return a *new* StepResult to replace the result, or None to keep
        the original.
        """
        ...


# ---------------------------------------------------------------------------
# HookChain — compose multiple hooks
# ---------------------------------------------------------------------------


class HookChain:
    """Ordered chain of hooks, executed sequentially.

    Usage::

        chain = HookChain(step.pre_hooks + step.post_hooks)
        step = chain.run_pre_hooks(step, ctx)
        result = executor.execute(step, context)
        result = chain.run_post_hooks(step, result, ctx)
    """

    def __init__(self, hooks: list[StepHook]) -> None:
        self._hooks = hooks

    @property
    def hooks(self) -> list[StepHook]:
        """The hooks in this chain."""
        return list(self._hooks)

    def run_pre_hooks(self, step: StepSpec, context: HookContext) -> StepSpec:
        """Run all pre-hooks in order. Each may return a replacement step."""
        for hook in self._hooks:
            if hasattr(hook, "on_pre_step"):
                try:
                    replacement = hook.on_pre_step(step, context)
                except Exception:
                    logger.exception(
                        "Pre-hook '%s' raised for step '%s'", hook.name, step.id
                    )
                    continue
                if replacement is not None:
                    step = replacement
        return step

    def run_post_hooks(
        self,
        step: StepSpec,
        result: StepResult,
        context: HookContext,
    ) -> StepResult:
        """Run all post-hooks in order. Each may return a replacement result."""
        for hook in self._hooks:
            if hasattr(hook, "on_post_step"):
                try:
                    replacement = hook.on_post_step(step, result, context)
                except Exception:
                    logger.exception(
                        "Post-hook '%s' raised for step '%s'", hook.name, step.id
                    )
                    continue
                if replacement is not None:
                    result = replacement
        return result


# ---------------------------------------------------------------------------
# HookFactoryContext — rebuild inputs passed to hook factories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookFactoryContext:
    """Rebuild inputs passed to hook factories at construction time.

    Carries the pipeline's step executor (nullable) so context-dependent
    hooks (e.g. ``RecoveryHook``) can be reconstructed by name.  Read-only
    scenarios never construct hooks, so they need no executor.
    """

    step_executor: Any | None = None
    # future: pipeline / domain handles, added without breaking factories.


# ---------------------------------------------------------------------------
# HookRegistry — discover and register hooks by name
# ---------------------------------------------------------------------------


class HookRegistry:
    """Registry of named hook factories.

    Hooks are registered by name so they can be referenced in YAML configs.

    Factories take a :class:`HookFactoryContext` (phase 5 — decision 33), so
    context-dependent hooks (e.g. ``RecoveryHook``, which needs the pipeline's
    ``step_executor``) can be rebuilt by name.  Zero-arg hooks adapt by
    accepting and ignoring the context.

    Usage::

        registry = HookRegistry()
        registry.register("audit_log", lambda ctx: AuditLogHook())
        registry.register("prompt_inject", lambda ctx: PromptInjectHook())

        hook = registry.create("audit_log", HookFactoryContext())
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[HookFactoryContext], StepHook]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[HookFactoryContext], StepHook],
    ) -> None:
        """Register a hook factory under *name*."""
        if name in self._factories:
            logger.warning("HookRegistry: overwriting registration for '%s'", name)
        self._factories[name] = factory

    def create(
        self,
        name: str,
        context: HookFactoryContext | None = None,
    ) -> StepHook:
        """Create a hook instance by *name*.

        Args:
            name: The registered hook name.
            context: Rebuild inputs passed to the factory (defaults to an
                empty context when omitted).

        Raises:
            KeyError: If *name* is not registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"Unknown hook: '{name}'. Registered: {sorted(self._factories)}")
        return factory(context or HookFactoryContext())

    def create_all(
        self,
        names: list[str],
        context: HookFactoryContext | None = None,
    ) -> list[StepHook]:
        """Create hook instances for each name in *names*."""
        return [self.create(n, context) for n in names]

    def create_from_configs(
        self,
        configs: list[dict[str, Any]],
        context: HookFactoryContext | None = None,
    ) -> list[StepHook]:
        """Create hooks from a list of config dicts with ``name`` and optional
        ``config`` keys.

        Example::

            configs = [
                {"name": "audit_log"},
                {"name": "prompt_inject", "config": {"extra_instructions": "..."}},
            ]
            hooks = registry.create_from_configs(configs)
        """
        hooks: list[StepHook] = []
        for cfg in configs:
            name = cfg["name"]
            hook_config = cfg.get("config", {})
            hook = self.create(name, context)
            for key, value in hook_config.items():
                if hasattr(hook, key):
                    setattr(hook, key, value)
            hooks.append(hook)
        return hooks

    def list_names(self) -> list[str]:
        """Return all registered hook names, sorted."""
        return sorted(self._factories)

    def discover_plugins(self) -> None:
        """Load hooks registered via setuptools entry points.

        Scans the ``quro.hooks`` entry-point group.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return

        for ep in entry_points(group="quro.hooks"):
            try:
                factory = ep.load()
                self.register(ep.name, factory)
                logger.debug("HookRegistry: loaded plugin '%s'", ep.name)
            except Exception:
                logger.exception(
                    "HookRegistry: failed to load plugin '%s'", ep.name
                )


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_default_registry: HookRegistry | None = None


def get_hook_registry() -> HookRegistry:
    """Return the global (singleton) HookRegistry.

    Lazily creates the registry and discovers plugins on first access.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = HookRegistry()
        _default_registry.discover_plugins()
    return _default_registry
