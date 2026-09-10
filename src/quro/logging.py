"""Unified logging configuration for quro (RC-7).

RC-7 of the L1 session-integrity investigation: every module declares
``logger = logging.getLogger(__name__)`` but no handler or level is ever
configured, so all INFO/DEBUG messages are swallowed by Python's
last-resort handler.  This module wires the stdlib ``logging`` framework
(no new wheel) from ``QURO_DEBUG`` / ``QURO_LOG_LEVEL``, and provides a
debug-only dump helper for the assembled MetaPlanner blocks.

Level resolution order:
  1. explicit ``level`` argument to :func:`configure_logging`
  2. ``QURO_LOG_LEVEL`` environment variable
  3. ``QURO_DEBUG=1`` → DEBUG
  4. otherwise → no-op (leave stdlib logging untouched)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_configured = False

# ANSI dim/grey escape for optional debug rendering.
DIM = "\033[2m"
RESET = "\033[0m"


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    numeric = getattr(logging, name, None)
    if isinstance(numeric, int):
        return numeric
    return logging.WARNING


def configure_logging(level: int | str | None = None) -> None:
    """Configure the root ``quro`` logger (idempotent).

    Without an explicit level, ``QURO_DEBUG`` or ``QURO_LOG_LEVEL`` this
    is a **no-op**: it leaves the stdlib logging state untouched so
    callers (and pytest's ``caplog``) keep their default propagation
    behaviour.  Only when debug logging is actually requested do we attach
    a stderr handler and set the ``quro`` logger level.
    """
    global _configured
    if _configured:
        return

    resolved: int | None
    if level is not None:
        resolved = _coerce_level(level)
    else:
        env_level = os.environ.get("QURO_LOG_LEVEL")
        if env_level:
            resolved = _coerce_level(env_level)
        elif os.environ.get("QURO_DEBUG") in ("1", "true", "True", "yes"):
            resolved = logging.DEBUG
        else:
            resolved = None

    if resolved is None:
        # No explicit request — leave stdlib logging untouched.
        _configured = True
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    ))

    logger = logging.getLogger("quro")
    logger.setLevel(resolved)
    logger.addHandler(handler)
    # NOTE: keep ``propagate=True`` so messages still reach the root logger
    # (pytest's caplog attaches there).  At DEBUG/INFO this may duplicate
    # WARNING+ lines on stderr, which is acceptable for opt-in debug runs.
    _configured = True


def is_debug_enabled() -> bool:
    """True when the ``quro`` logger is at DEBUG level."""
    return logging.getLogger("quro").isEnabledFor(logging.DEBUG)


def is_dump_context_enabled() -> bool:
    """True when assembled prompt blocks should be dumped to stderr.

    Active when either ``QURO_DEBUG`` or ``QURO_DUMP_CONTEXT`` is set.
    """
    if is_debug_enabled():
        return True
    import os
    return os.environ.get("QURO_DUMP_CONTEXT") in ("1", "true", "True", "yes")


def dim(text: str) -> str:
    """Wrap *text* in a dim/grey ANSI escape for terminal output."""
    return f"{DIM}{text}{RESET}"


def dump_assembled_blocks(
    blocks: dict[str, str] | None = None,
    full_context_list: list[dict[str, Any]] | None = None,
) -> str:
    """Render the assembled MetaPlanner blocks / message list for debug.

    Returns a plain-text dump; the caller decides whether to log it
    (gated on ``is_debug_enabled()``).  Returns ``""`` when nothing is
    provided.
    """
    parts: list[str] = []
    if blocks:
        parts.append("── Assembled blocks ──")
        for name, content in blocks.items():
            parts.append(f"[{name}]\n{content}")
    if full_context_list:
        parts.append("── Assembled message list ──")
        for i, m in enumerate(full_context_list):
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            parts.append(f"[{i}] {role}: {content}")
    return "\n\n".join(parts)
