"""Context Layer protocols.

Core ABCs / Protocols that define the contract for the unified context
management orchestration layer.  All consumers depend on these interfaces —
never on concrete implementations.

Three axes, three owners (Phase 0 — unified resource layer):

- **IViewProjector** — pure, stateless block projection (view layer)
- **IPromptCoordinator** — pure ordering + block-level trim (coordinator)
- **ISessionContext** — stateful session runtime, scope narrowed to the
  conversation-history window (context layer)
- **IPromptBlockProvider** — prompt block structure provider (view layer)
- **SessionHook** — lifecycle hook (Protocol, structural subtyping)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

from quro.context.models import (
    BlockTree,
    SessionContextConfig,
    SessionContextReport,
    TokenLedger,
)


# ===========================================================================
# ISessionContext — session-level context orchestrator
# ===========================================================================


class ISessionContext(ABC):
    """Session-level context orchestrator — the ONLY stateful home.

    Scope narrowed (Phase 0): it assembles blocks into a message list and
    keeps the conversation-history window.  **Trim is conversation-history
    only** — it does not know block kinds (block-level trim lives in the
    coordinator, by ``KIND_ORDER`` priority).

    Typical usage::

        ctx = DefaultSessionContext()
        ctx.open_session(config)
        for round_idx in range(max_rounds):
            messages = ctx.assemble(
                system_blocks={"role": "...", "constraints": "..."},
                user_blocks={"problem": "...", "projected": "..."},
                conversation=history,
            )
            response = backend.complete_one(messages)
            ctx.update(round_index=round_idx, response=response)
        report = ctx.close_session()
        print(report.format())
    """

    # --- lifecycle -------------------------------------------------------

    @abstractmethod
    def open_session(self, config: SessionContextConfig) -> None:
        """Open a new session.

        Initialises the block tree and token ledger.  Triggers
        ``pre_session`` hooks.

        Args:
            config: Session configuration (block layout, capacity, trim policy).
        """
        ...

    @abstractmethod
    def close_session(self) -> SessionContextReport:
        """Close the session and return a final report.

        Triggers ``post_session`` hooks.
        """
        ...

    # --- assembly --------------------------------------------------------

    @abstractmethod
    def assemble(
        self,
        *,
        system_blocks: dict[str, str] | None = None,
        user_blocks: dict[str, str] | None = None,
        conversation: list[BaseMessage] | None = None,
    ) -> list[BaseMessage]:
        """Assemble block content into a LangChain message list.

        Triggers ``in_session`` hooks (assemble phase).

        Each returned message carries block-attribution metadata so
        downstream consumers can trace which block a message originated
        from.

        Args:
            system_blocks: Mapping of block name → content for system blocks.
            user_blocks: Mapping of block name → content for user blocks.
            conversation: Existing conversation history to append.

        Returns:
            A list of :class:`~langchain_core.messages.BaseMessage` ready
            for the LLM backend.
        """
        ...

    @abstractmethod
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
        into the token ledger, attributing prompt tokens proportionally
        across blocks.

        Args:
            round_index: The round being recorded.
            response: The assistant message returned by the LLM.
            tool_results: Optional tool-result messages from the same round.
        """
        ...

    # --- queries ---------------------------------------------------------

    @abstractmethod
    def get_ledger(self) -> TokenLedger:
        """Return the current token ledger."""
        ...

    @abstractmethod
    def get_block_tree(self) -> BlockTree:
        """Return the current block tree."""
        ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of session context state.

        Useful for pause / resume workflows.
        """
        ...


# ===========================================================================
# SessionHook — lifecycle hook (Protocol)
# ===========================================================================


@runtime_checkable
class SessionHook(Protocol):
    """Session lifecycle hook.

    Three hook points map to the session context lifecycle:

    - ``pre_session``  — after :meth:`ISessionContext.open_session`, before first ``assemble``
    - ``in_session``   — before each ``assemble`` / after each ``update``
    - ``post_session`` — before :meth:`ISessionContext.close_session` returns

    Hooks are composable: multiple hooks can be registered and execute in
    registration order.

    .. note::

        This is a :class:`~typing.Protocol` — any object with matching
        method signatures satisfies the interface (structural subtyping).
    """

    name: str
    """Human-readable hook name (used in logs and registries)."""

    def on_pre_session(self, ctx: ISessionContext) -> None:
        """Called after session open, before the first ``assemble``.

        Use to initialise block structure, inject static content, etc.
        """
        ...

    def on_in_session_assemble(self, ctx: ISessionContext) -> None:
        """Called before each ``assemble()`` invocation.

        Use to update block content dynamically (e.g. refresh projected
        state, inject latest feedback).
        """
        ...

    def on_in_session_update(self, ctx: ISessionContext) -> None:
        """Called after each ``update()`` invocation.

        Use to inspect token consumption, trigger trimming, or persist
        intermediate state.
        """
        ...

    def on_post_session(self, ctx: ISessionContext) -> None:
        """Called before ``close_session()`` returns.

        Use to persist final state, emit reports, or archive.
        """
        ...


# ===========================================================================
# IPromptBlockProvider — prompt block structure
# ===========================================================================


class IPromptBlockProvider(ABC):
    """Expose prompt content as named blocks.

    Implementations may be:

    - ``StaticBlockProvider`` (hard-coded blocks for testing / simple cases)
    - ``RemoteBlockProvider`` (blocks fetched from an external service)

    Block names are bare (``problem``, ``hints``, ``objective``) — the
    ``system.role`` / ``user.projected`` path-naming is retired (Phase 0).
    """

    @abstractmethod
    def get_blocks(self, principal: str) -> dict[str, str]:
        """Return all blocks for *principal* (a step-type / session name).

        Args:
            principal: The requesting identity (e.g. ``"meta-planner"``).

        Returns:
            Mapping of ``{block_name: content}``, e.g.
            ``{"role": "You are...", "constraints": "JUST CREATE..."}``.
        """
        ...

    @abstractmethod
    def list_principals(self) -> list[str]:
        """List all available principal names."""
        ...

    @abstractmethod
    def list_blocks(self, principal: str) -> list[str]:
        """List all block names for *principal*.

        Args:
            principal: A requesting identity.

        Returns:
            Block names in declaration order.
        """
        ...


# ===========================================================================
# IViewProjector — pure, stateless block projection (Phase 0 skeleton)
# ===========================================================================


class IViewProjector(Protocol):
    """Project a step's visible context as named blocks.

    Pure and stateless: the same ``(state, phase, principal)`` always yields
    the same blocks.  ``visible_blocks`` is the ACL predicate — default-deny
    plus ownership/dependency/capability grants — rendered at assembly time,
    never hard-coded by an identity class.

    Phase 0 lands the signature only; the ACL grants and deny logic arrive in
    the resource-layer stages.
    """

    def render_block(
        self,
        name: str,
        state: dict[str, Any],
        *,
        phase: str,
        budget: int,
        principal: Any,
    ) -> str:
        """Render one block's content for *principal* in *phase*."""
        ...

    def visible_blocks(self, principal: Any, phase: str) -> set[str]:
        """Return the set of block names *principal* may see in *phase*."""
        ...


# ===========================================================================
# IPromptCoordinator — pure ordering + block-level trim (Phase 0 skeleton)
# ===========================================================================


class IPromptCoordinator(Protocol):
    """Collect, order, and trim blocks into system/user containers.

    Pure ordering + block-level trim: the coordinator owns the ``KIND_ORDER``
    authority (a framework constant the domain cannot override) and trims by
    kind priority.  ``@phase_blocks`` is a participation set only — which
    blocks compose a phase — never an ordering authority.
    """

    def assemble(
        self,
        *,
        state: dict[str, Any],
        principal: Any,
        phase: str,
        projector: IViewProjector,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return ``(system_blocks, user_blocks)`` in cognitive order.

        Collect → order (``KIND_ORDER``) → trim by kind priority →
        return the two containers.
        """
        ...
