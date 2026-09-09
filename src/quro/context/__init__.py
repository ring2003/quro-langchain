"""Context Layer — unified context management orchestration.

Manages the organisation, assembly, trimming, and token accounting of all
content sent to the LLM within a session.  Protocol-driven: all consumers
depend on ABC interfaces (``ISessionContext``, ``IViewProjector``,
``IPromptCoordinator``, ``IPromptBlockProvider``), never on concrete
implementations.

Package contents:

- :mod:`quro.context.protocols` — ``ISessionContext``, ``IViewProjector``,
  ``IPromptCoordinator``, ``IPromptBlockProvider``, ``SessionHook``
- :mod:`quro.context.models` — ``ContextBlock``, ``BlockTree``, ``TokenLedger``, ``SessionContextConfig``, etc.
- :mod:`quro.context.default_session` — ``DefaultSessionContext`` (Phase 1 concrete implementation)
"""

from quro.context.default_session import DefaultSessionContext
from quro.context.models import (
    BlockTokenReport,
    BlockTree,
    ContextBlock,
    LedgerEntry,
    LedgerSummary,
    RoundTokenReport,
    SessionContextConfig,
    SessionContextReport,
    TokenLedger,
    TrimAction,
)
from quro.context.protocols import (
    IPromptBlockProvider,
    IPromptCoordinator,
    ISessionContext,
    IViewProjector,
    SessionHook,
)

__all__ = [
    # Protocols
    "IPromptBlockProvider",
    "IPromptCoordinator",
    "ISessionContext",
    "IViewProjector",
    "SessionHook",
    # Implementation
    "DefaultSessionContext",
    # Models
    "BlockTokenReport",
    "BlockTree",
    "ContextBlock",
    "LedgerEntry",
    "LedgerSummary",
    "RoundTokenReport",
    "SessionContextConfig",
    "SessionContextReport",
    "TokenLedger",
    "TrimAction",
]
