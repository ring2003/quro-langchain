"""Step-driven tool registry and resolver.

Phase 0 cleanup: step-driven registry/resolver.  Tool allocation is derived
from a step's explicit ``StepSpec.tools`` and its ``step_type`` (skill pool +
phase) upstream — there is no capability/role expansion table.

The ``ToolRegistry`` holds two static sources of tools:

- **worker tools**: static filesystem tools (``read``, ``write``, ...).
- **mcp tools**: tools loaded from MCP servers, optionally tagged with the
  server name that provided them.

Per-session *domain* tools (created by ``make_all_tools``) are not
registered here — they are phase-specific and supplied by the caller.
"""

from __future__ import annotations

from typing import Any, Iterable

from quro.core.worker_tools import all_worker_tools
from quro.steps.core import StepSpec


def _dedup(tools: Iterable[Any]) -> list[Any]:
    """De-duplicate tools by name, preserving first-seen order."""
    seen: set[str] = set()
    result: list[Any] = []
    for t in tools:
        name = getattr(t, "name", "")
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        result.append(t)
    return result


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of tool pools with name-based lookup.

    Holds tools from two static sources:

    - ``worker_tools``: filesystem tools (``read``, ``write``, ``ls``,
      ``edit``, ``grep``, ``find``, ``shell``).
    - ``mcp_tools``: tools loaded from MCP servers. Each tool may be tagged
      with the server name that provided it via ``mcp_server_map``.

    Args:
        worker_tools: Static filesystem tool objects.
        mcp_tools: Flat list of MCP tool objects.
        mcp_server_map: Mapping of server name to the tools it provided.
    """

    def __init__(
        self,
        worker_tools: Iterable[Any] | None = None,
        mcp_tools: Iterable[Any] | None = None,
        mcp_server_map: dict[str, list[Any]] | None = None,
    ) -> None:
        self._worker_tools: list[Any] = list(worker_tools or [])
        self._mcp_tools: list[Any] = list(mcp_tools or [])
        self._mcp_server_map: dict[str, list[Any]] = dict(mcp_server_map or {})
        self._by_name: dict[str, Any] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_name = {}
        for t in self._worker_tools + self._mcp_tools:
            name = getattr(t, "name", "")
            if name and name not in self._by_name:
                self._by_name[name] = t

    @property
    def worker_tools(self) -> list[Any]:
        """All registered worker (filesystem) tools."""
        return list(self._worker_tools)

    @property
    def mcp_tools(self) -> list[Any]:
        """All registered MCP tools."""
        return list(self._mcp_tools)

    @property
    def server_names(self) -> list[str]:
        """Names of all MCP servers that contributed tools."""
        return sorted(self._mcp_server_map)

    def register_worker(self, tool: Any) -> None:
        """Register a worker (filesystem) tool."""
        self._worker_tools.append(tool)
        self._rebuild_index()

    def register_mcp(self, tool: Any, server: str | None = None) -> None:
        """Register an MCP tool, optionally tagged with its server name."""
        self._mcp_tools.append(tool)
        if server:
            self._mcp_server_map.setdefault(server, []).append(tool)
        self._rebuild_index()

    def has(self, name: str) -> bool:
        """Return True if a tool with *name* is registered."""
        return name in self._by_name

    def get(self, name: str) -> Any | None:
        """Return the registered tool with *name*, or None."""
        return self._by_name.get(name)

    def by_names(self, names: Iterable[str]) -> list[Any]:
        """Return registered tools whose names are in *names* (ordered, deduped)."""
        wanted = set(names)
        return _dedup(t for t in self._worker_tools + self._mcp_tools if t.name in wanted)

    def by_servers(self, servers: Iterable[str]) -> list[Any]:
        """Return the MCP tools provided by the given server names."""
        result: list[Any] = []
        for server in servers:
            result.extend(self._mcp_server_map.get(server, []))
        return _dedup(result)


# ---------------------------------------------------------------------------
# ToolResolver
# ---------------------------------------------------------------------------


class ToolResolver:
    """Binds steps to concrete tools via a ``ToolRegistry``.

    Step-driven: the step declares its external tools explicitly
    (``StepSpec.tools``); MCP tools come from the registry's full MCP pool.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """The backing ``ToolRegistry``."""
        return self._registry

    def filter(self, tools: Iterable[Any], names: Iterable[str] | None) -> list[Any]:
        """Keep only tools whose name is in *names*; empty/None keeps all."""
        tools = list(tools)
        if not names:
            return tools
        wanted = set(names)
        return [t for t in tools if t.name in wanted]

    def resolve_for_step(
        self,
        step: StepSpec,
        step_types: Any | None = None,
    ) -> list[Any]:
        """Resolve the external tools (worker + MCP) for a step.

        Worker tools come from ``step.tools``.  When ``step.tools`` is empty
        and a ``StepTypeCatalog`` is provided, tools are resolved from the
        step's ``step_type`` (runtime resolution for ``create_step`` outputs).
        MCP tools come from the registry's full MCP pool (server-level grants
        move to the resource-layer ACL later).
        """
        tools = step.tools
        if not tools and step_types and step.step_type:
            tools = step_types.tools_for(step.step_type)
        worker = self._registry.by_names(tools)
        mcp = list(self._registry.mcp_tools)
        return _dedup(worker + mcp)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_default_registry(
    mcp_tools: Iterable[Any] | None = None,
    mcp_server_map: dict[str, list[Any]] | None = None,
) -> ToolRegistry:
    """Build a ``ToolRegistry`` preloaded with the default worker pools.

    Args:
        mcp_tools: Flat list of MCP tool objects to register.
        mcp_server_map: Mapping of MCP server name to its tools.

    Returns:
        A ``ToolRegistry`` with explorer + builder worker tools and the
        given MCP tools.
    """
    worker_tools = all_worker_tools()
    return ToolRegistry(
        worker_tools=worker_tools,
        mcp_tools=mcp_tools,
        mcp_server_map=mcp_server_map,
    )
