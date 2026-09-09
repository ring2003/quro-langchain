"""Tool filter hook — dynamically adjust the tool set for a step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from quro.steps.hooks import HookContext

if TYPE_CHECKING:
    from quro.steps.core import StepResult, StepSpec


@dataclass
class ToolFilterHook:
    """Dynamically add or remove tools for a step.

    Usage in YAML::

        pre_hooks:
          - name: tool_filter
            config:
              add_tools: [webfetch]
              remove_tools: [shell]

    Args:
        add_tools: Tool names to make available (in addition to declared tools).
        remove_tools: Tool names to exclude from this step.
    """

    name: str = "tool_filter"
    add_tools: list[str] = field(default_factory=list)
    remove_tools: list[str] = field(default_factory=list)

    def on_pre_step(
        self, step: StepSpec, context: HookContext
    ) -> None:
        context.extra_tools.extend(self.add_tools)
        context.remove_tools.update(self.remove_tools)
