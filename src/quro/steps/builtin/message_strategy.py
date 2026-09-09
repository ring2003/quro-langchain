"""Message strategy hook — controls cross-round message management per step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quro.steps.hooks import HookContext

if TYPE_CHECKING:
    from quro.steps.core import StepSpec


@dataclass
class MessageStrategyHook:
    """Set the message strategy for a step via ``HookContext.metadata``.

    Usage in YAML::

        pre_hooks:
          - name: message_strategy
            config:
              strategy: accumulate

    Args:
        strategy: ``"rebuild"`` or ``"accumulate"``.
        state_injection_interval: In ACCUMULATE mode, inject fresh projected
            state every N idle rounds (default 2).
        max_budget_chars: Soft cap on total message content length
            (0 = no cap).
    """

    name: str = "message_strategy"
    strategy: str = "accumulate"
    state_injection_interval: int = 2
    max_budget_chars: int = 0

    def on_pre_step(
        self, step: StepSpec, context: HookContext
    ) -> None:
        context.metadata["message_strategy"] = self.strategy
        context.metadata["message_state_injection_interval"] = (
            self.state_injection_interval
        )
        context.metadata["message_max_budget_chars"] = self.max_budget_chars
