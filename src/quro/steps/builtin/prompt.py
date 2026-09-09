"""Prompt inject hook — inject extra instructions into the system prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from quro.steps.hooks import HookContext

if TYPE_CHECKING:
    from quro.steps.core import StepResult, StepSpec


@dataclass
class PromptInjectHook:
    """Inject additional instructions into the system prompt.

    Usage in YAML::

        pre_hooks:
          - name: prompt_inject
            config:
              extra_instructions: "Use type hints for all functions."

    Args:
        extra_instructions: Text to append to the system prompt.
        prepend: If True, prepend instead of append.
    """

    name: str = "prompt_inject"
    extra_instructions: str = ""
    prepend: bool = False

    def on_pre_step(
        self, step: StepSpec, context: HookContext
    ) -> None:
        if not self.extra_instructions:
            return
        existing = context.system_prompt or ""
        if self.prepend:
            context.system_prompt = f"{self.extra_instructions}\n\n{existing}"
        else:
            context.system_prompt = f"{existing}\n\n{self.extra_instructions}"
