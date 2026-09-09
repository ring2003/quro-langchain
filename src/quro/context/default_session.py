"""DefaultSessionContext — concrete implementation of ISessionContext.

Wraps ``BlockTree`` and ``TokenLedger`` to provide the unified context
management orchestration defined in ``ISessionContext``.

Phase 1 implementation: block-aware message assembly with token estimation,
trim-on-overflow, and per-block/per-round accounting.

Usage with ExecutionGovernor::

    ctx = DefaultSessionContext()
    ctx.open_session(SessionContextConfig())

    for round_idx in range(max_rounds):
        messages = ctx.assemble(
            system_blocks={"role": role_prompt, "tools": tools_desc},
            user_blocks={"problem": projected_state},
        )
        response = backend.complete_one(messages, tools=tools)
        ctx.update(round_index=round_idx, response=response)

    report = ctx.close_session()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from quro.context.models import (
    BlockTree,
    SessionContextConfig,
    SessionContextReport,
    TokenLedger,
)
from quro.context.protocols import ISessionContext, SessionHook

logger = logging.getLogger(__name__)

# Rough character→token estimator (~4 chars per token for English text).
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


class DefaultSessionContext(ISessionContext):
    """Concrete session-level context orchestrator.

    Manages block tree, token ledger, and lifecycle hooks.  Compatible
    with any ``ISessionContext`` consumer — Governor, adapter, or
    pipeline runner.
    """

    def __init__(self) -> None:
        self._config: SessionContextConfig | None = None
        self._block_tree: BlockTree | None = None
        self._ledger: TokenLedger | None = None
        self._hooks: list[SessionHook] = []
        self._round_index: int = 0
        self._trim_count: int = 0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_session(self, config: SessionContextConfig) -> None:
        """Open a new session.

        Initialises the block tree and token ledger.  Triggers
        ``pre_session`` hooks.
        """
        self._config = config
        self._block_tree = BlockTree()
        self._ledger = TokenLedger()
        self._round_index = 0
        self._trim_count = 0
        self._opened = True

        for hook in self._hooks:
            try:
                hook.on_pre_session(self)
            except Exception:
                logger.debug("Hook %s.on_pre_session failed", getattr(hook, "name", "?"),
                             exc_info=True)

    def close_session(self) -> SessionContextReport:
        """Close the session and return a final report.

        Triggers ``post_session`` hooks.
        """
        for hook in self._hooks:
            try:
                hook.on_post_session(self)
            except Exception:
                logger.debug("Hook %s.on_post_session failed", getattr(hook, "name", "?"),
                             exc_info=True)

        self._opened = False
        ledger = self._ledger or TokenLedger()
        return SessionContextReport(
            total_rounds=self._round_index,
            ledger_summary=ledger.summarize(),
            trim_events=self._trim_count,
            metadata={
                "budget": self._config.max_tokens if self._config else 0,
            },
        )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        *,
        system_blocks: dict[str, str] | None = None,
        user_blocks: dict[str, str] | None = None,
        conversation: list[BaseMessage] | None = None,
    ) -> list[BaseMessage]:
        """Assemble block content into a LangChain message list.

        Triggers ``in_session`` hooks (assemble phase).

        Each block is rendered as a separate message with block-attribution
        metadata so downstream consumers can trace which block a message
        originated from.

        Args:
            system_blocks: Mapping of block name → content for system blocks.
            user_blocks: Mapping of block name → content for user blocks.
            conversation: Existing conversation history to append.

        Returns:
            A list of :class:`~langchain_core.messages.BaseMessage` ready
            for the LLM backend.
        """
        self._ensure_opened()

        for hook in self._hooks:
            try:
                hook.on_in_session_assemble(self)
            except Exception:
                logger.debug("Hook %s.on_in_session_assemble failed",
                             getattr(hook, "name", "?"), exc_info=True)

        messages: list[BaseMessage] = []

        # --- System blocks ---
        if system_blocks:
            tree = self._block_tree
            for block_name, content in system_blocks.items():
                if not content:
                    continue
                path = f"system.{block_name}"
                tree.add_block(path, content)
                messages.append(
                    SystemMessage(
                        content=content,
                        additional_kwargs={"block_path": path},
                    )
                )

        # --- User blocks ---
        if user_blocks:
            tree = self._block_tree
            for block_name, content in user_blocks.items():
                if not content:
                    continue
                path = f"user.{block_name}"
                tree.add_block(path, content)
                messages.append(
                    HumanMessage(
                        content=content,
                        additional_kwargs={"block_path": path},
                    )
                )

        # --- Append conversation history ---
        if conversation:
            messages.extend(conversation)

        # --- Token estimation ---
        round_idx = self._round_index
        for msg in messages:
            path = msg.additional_kwargs.get("block_path", "conversation")
            est = _estimate_tokens(str(msg.content))
            self._ledger.record_estimate(path, round_idx, est)

        # --- Trim if over budget ---
        self._trim_if_needed(messages)

        return messages

    def assemble_dicts(
        self,
        *,
        system_blocks: dict[str, str] | None = None,
        user_blocks: dict[str, str] | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble block content into a dict-based message list.

        Convenience wrapper around ``assemble()`` that returns plain
        ``dict`` messages compatible with ``IRuntimeBackend.complete_one``.

        Args:
            system_blocks: Mapping of block name → content for system blocks.
            user_blocks: Mapping of block name → content for user blocks.
            conversation: Existing conversation history to append (dict format).

        Returns:
            A list of ``dict`` messages with ``role`` and ``content`` keys.
        """
        langchain_msgs = self.assemble(
            system_blocks=system_blocks,
            user_blocks=user_blocks,
        )
        result: list[dict[str, Any]] = []
        for msg in langchain_msgs:
            role = "system" if isinstance(msg, SystemMessage) else "user"
            entry: dict[str, Any] = {
                "role": role,
                "content": str(msg.content),
            }
            block_path = msg.additional_kwargs.get("block_path")
            if block_path:
                entry["_block_path"] = block_path
            result.append(entry)

        if conversation:
            result.extend(conversation)

        return result

    def update(
        self,
        *,
        round_index: int,
        response: BaseMessage,
        tool_results: list[BaseMessage] | None = None,
    ) -> None:
        """Record state after one LLM interaction round.

        Triggers ``in_session`` hooks (update phase).

        Back-fills actual token counts from ``response.usage_metadata``
        into the token ledger.
        """
        self._ensure_opened()
        self._round_index = max(self._round_index, round_index + 1)

        # Back-fill actual token counts.
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        if input_tokens or output_tokens:
            self._ledger.record_actual(
                round_index=round_index,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        for hook in self._hooks:
            try:
                hook.on_in_session_update(self)
            except Exception:
                logger.debug("Hook %s.on_in_session_update failed",
                             getattr(hook, "name", "?"), exc_info=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_ledger(self) -> TokenLedger:
        """Return the current token ledger."""
        self._ensure_opened()
        return self._ledger

    def get_block_tree(self) -> BlockTree:
        """Return the current block tree."""
        self._ensure_opened()
        return self._block_tree

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of session context state."""
        self._ensure_opened()
        return {
            "round_index": self._round_index,
            "trim_count": self._trim_count,
            "ledger_summary": self._ledger.summarize(),
            "config": {
                "max_tokens": self._config.max_tokens if self._config else 0,
                "trim_strategy": self._config.trim_strategy if self._config else "last",
            },
        }

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook: SessionHook) -> None:
        """Register a lifecycle hook."""
        self._hooks.append(hook)

    def remove_hook(self, name: str) -> None:
        """Remove a hook by name."""
        self._hooks = [h for h in self._hooks if getattr(h, "name", "") != name]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_opened(self) -> None:
        if not self._opened:
            raise RuntimeError(
                "DefaultSessionContext: session not opened. "
                "Call open_session() before assemble() or update()."
            )

    def _trim_if_needed(self, messages: list[BaseMessage]) -> None:
        """Trim oldest non-system messages when approaching token budget."""
        if self._config is None or self._config.max_tokens <= 0:
            return

        total_est = sum(_estimate_tokens(str(m.content)) for m in messages)
        threshold = int(self._config.max_tokens * self._config.compression_threshold)

        if total_est <= threshold:
            return

        self._trim_count += 1

        # Keep system messages + N most recent user/assistant/tool messages.
        keep_tail = 8
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= keep_tail:
            return

        trimmed_others = other_msgs[-keep_tail:]
        messages[:] = (
            system_msgs
            + [HumanMessage(
                content="[Earlier interactions trimmed for brevity]",
                additional_kwargs={"block_path": "system.trim_marker"},
            )]
            + trimmed_others
        )

        logger.debug(
            "Trimmed messages: %d → %d (budget %d, threshold %d, est %d)",
            len(other_msgs), len(trimmed_others),
            self._config.max_tokens, threshold, total_est,
        )
