"""PromptAssemblyPolicy — builds system prompt and manages context budget.

Fixes R§7 (message rebuild) by injecting session digest and applying
budget ordering (HINTS/feedback first, not truncated first).

See ``docs/system-analysis/09-refactoring-implementation-plan.md`` §4.4.
"""

from __future__ import annotations

from typing import Any

from quro.steps.hooks import HookContext


class PromptAssemblyPolicy:
    """Protocol for prompt assembly."""

    def build_system(
        self,
        *,
        state: dict[str, Any],
        hook_context: HookContext | None,
    ) -> str:
        ...


class DefaultPromptAssemblyPolicy:
    """Build system prompt with dynamic state variables.

    Uses the step type's template when available (CONTINUE MODE / RE-ENTRY
    MODE vars), falling back to the static identity block.  Hook overrides are
    respected.
    """

    def __init__(
        self,
        *,
        context_budget_chars: int = 12000,
        template_loader: Any = None,
        role_name: str | None = None,
        fallback_system_prompt: str = "",
    ) -> None:
        self.context_budget_chars = context_budget_chars
        self.template_loader = template_loader
        self.role_name = role_name
        self.fallback_system_prompt = fallback_system_prompt

    def build_system(
        self,
        *,
        state: dict[str, Any],
        hook_context: HookContext | None,
    ) -> str:
        # Hook override takes priority
        if hook_context is not None and hook_context.system_prompt is not None:
            return hook_context.system_prompt

        # Template-based rendering with dynamic variables
        if self.template_loader is not None and self.role_name:
            pending = [s for s in state.get("steps", [])
                       if s.get("status") in ("pending", "in_progress")]
            completed = [s for s in state.get("steps", [])
                         if s.get("status") == "completed"]
            cancelled = [s for s in state.get("steps", [])
                         if s.get("status") == "cancelled"]
            try:
                return self.template_loader.render_system(
                    self.role_name,
                    has_pending_steps=len(pending) > 0,
                    pending_count=len(pending),
                    re_entry=len(completed) > 0 or len(cancelled) > 0,
                )
            except Exception:
                # Template not found — fall through to static prompt
                pass

        return self.fallback_system_prompt or ""
