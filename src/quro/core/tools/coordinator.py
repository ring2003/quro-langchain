"""ToolCoordinator — the single runtime authority for tool governance.

The tool counterpart of ``PromptCoordinator``: it owns
``collect → resolve → allocate → bind → invoke`` for one step's tool surface,
collapsing the previously scattered allocation and invocation sites into one
protocol-driven component.

Design: ``docs/unsat-policy-loop/architecture/tool-coordinator.md`` (decisions
40–46).  Allocation is **default-deny + grant**:

    tool surface = contract(principal, features, depends_on)
                 + lifecycle(principal)
                 + grants(StepType.tools)
                 + mcp (server-level, always-on until the ACL owns it)

The coordinator is pure and session-bound: it holds the name → handler index
and the allocation constants, but persists nothing.  Serialization is
``StepConfig``'s job, never the coordinator's.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quro.core.features import Feature
from quro.core.tools.capability import is_capability, tools_for_capabilities


# ---------------------------------------------------------------------------
# Framework tool-name lists — the allocation authority's constants.
#
# These mirror the prompt layer's ``KIND_ORDER``: a framework constant, not an
# author-facing declaration.  Order matters (it is the surface order).
# ---------------------------------------------------------------------------

# Injected on every worker step; the author cannot remove them.
CONTRACT_TOOLS: tuple[str, ...] = (
    "complete_step",
    "get_step",
    "terminate_subtree",
    "request_clarification",
    "checkpoint",
    "backtrack",
)

# Artifact read/write — always allocated to the step agent.  The READ
# (`get_artifact`) is unconditional at allocation time; per-artifact visibility
# is enforced by the ACL engine at invocation time (own-artifact and
# access="hints" dependency grants), NOT by a depends_on gate.  Previously
# `get_artifact` lived in the depends_on-gated DEPENDENCY_TOOLS, so a single
# self-steered explorer step without dependencies could never read its
# predecessor artifacts (bugreport-artifact RC-A).
ARTIFACT_TOOLS: tuple[str, ...] = ("add_artifact", "get_artifact")

# Injected only when the step carries ``Feature.RECOVERY``.
RECOVERY_TOOLS: tuple[str, ...] = ("commit_clue", "continue_step")

# Lifecycle tools — the planner session's contract (never a worker step).
PLANNER_TOOLS: tuple[str, ...] = (
    "set_interpretation",
    "set_plan",
    "confirm_understanding",
    "create_step",
    "update_step",
    "finalize_step",
    "resolve_clarification",
)

# Lifecycle tools — the evaluator session's contract.
EVALUATOR_TOOLS: tuple[str, ...] = ("submit_evaluation",)

# Principals (lifecycle roles).
PRINCIPAL_WORKER = "worker"
PRINCIPAL_PLANNER = "planner"
PRINCIPAL_EVALUATOR = "evaluator"


@runtime_checkable
class IToolCoordinator(Protocol):
    """Govern the tool surface of one step: collect → resolve → allocate → invoke.

    Pure and session-bound: it holds no persistent state and never writes to
    storage.
    """

    def surface(self, *, step: Any, phase: str, principal: str) -> list[Any]:
        """Return the concrete tools *step* may call in *phase* (allocate result)."""
        ...

    def invoke(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Route *tool_name* to its handler (the single invocation authority)."""
        ...


class ToolCoordinator:
    """Allocate a step's tool surface and route invocations.

    Args:
        internal_tools: Domain/framework semantic tools by name (always-on;
            their *semantic* availability is gated by the domain's ``apply``
            op table, not by the coordinator).
        external_tools: Filesystem grants by name (capability-resolved).
        mcp_tools: MCP tools (server-level, always-on).
        extra_tools: Always-on runtime tools (skill loaders, hook additions).
        remove_tools: Names removed by a tool-filter hook.
        mcp_session: Optional MCP session used to invoke MCP tools.
        step_types: Optional StepTypeCatalog for runtime resolution of tools
            from StepType when ``step.tools`` is empty (create_step fix).
    """

    def __init__(
        self,
        *,
        internal_tools: dict[str, Any],
        external_tools: dict[str, Any] | None = None,
        mcp_tools: list[Any] | None = None,
        extra_tools: list[Any] | None = None,
        remove_tools: set[str] | None = None,
        mcp_session: Any = None,
        step_types: Any | None = None,
    ) -> None:
        self._internal = dict(internal_tools)
        self._external = dict(external_tools or {})
        self._mcp_tools = list(mcp_tools or [])
        self._mcp_names = {getattr(t, "name", "") for t in self._mcp_tools}
        self._extra = list(extra_tools or [])
        self._remove = set(remove_tools or set())
        self._mcp_session = mcp_session
        self._step_types = step_types

    # ------------------------------------------------------------------
    # IToolCoordinator
    # ------------------------------------------------------------------

    def surface(
        self,
        *,
        step: Any,
        phase: str,
        principal: str,
    ) -> list[Any]:
        """Return the concrete tool surface for *step*.

        ``collect`` (contract + lifecycle + ``StepType.tools`` grants) →
        ``resolve`` (name → handler, fail-fast on a missing name) →
        ``allocate`` (default-deny; internal tools always-on, external grants
        always present, MCP + hook additions appended, removals dropped).

        ``phase`` is accepted for caller compatibility but no longer gates the
        surface: internal tool availability is enforced by the domain's own
        ``apply`` op table (``PHASE_MISMATCH``), so double-gating here was
        redundant.
        """
        names = self._collect(step, principal)

        result: list[Any] = []
        seen: set[str] = set()

        def _append(tool: Any) -> None:
            if tool.name in self._remove or tool.name in seen:
                return
            seen.add(tool.name)
            result.append(tool)

        for name in names:
            _append(self._resolve(name))
        for tool in self._extra:
            _append(tool)
        for tool in self._mcp_tools:
            _append(tool)

        return result

    def invoke(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Route *tool_name* to its handler (MCP or local).

        A name outside the index returns ``Unknown tool: …`` — never silently
        executes.
        """
        tool = self._resolve_optional(tool_name)
        if tool is None:
            return f"Unknown tool: {tool_name}"

        if tool_name in self._mcp_names and self._mcp_session is not None:
            try:
                return str(self._mcp_session.invoke_tool(tool_name, tool_args))
            except Exception as exc:  # pragma: no cover - defensive
                return f"Error invoking MCP tool '{tool_name}': {exc}"

        try:
            return str(tool.invoke(tool_args))
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # collect / resolve
    # ------------------------------------------------------------------

    def _collect(
        self,
        step: Any,
        principal: str,
    ) -> list[str]:
        """Collect the declared tool names for *step* (order-preserving, deduped)."""
        from quro.core.tools.grant import ToolProfile, evaluate_grant

        names = []

        def add(entries: tuple[str, ...]) -> None:
            for name in entries:
                if name not in names:
                    names.append(name)

        add(CONTRACT_TOOLS)
        add(ARTIFACT_TOOLS)
        if Feature.RECOVERY.value in step.features:
            add(RECOVERY_TOOLS)
        if principal == PRINCIPAL_PLANNER:
            add(PLANNER_TOOLS)
        elif principal == PRINCIPAL_EVALUATOR:
            add(EVALUATOR_TOOLS)

        # Resolve tools from StepType when step.tools is empty (create_step fix).
        # This ensures steps created via create_step get their StepType's tools.
        step_tools = step.tools
        if not step_tools and self._step_types and getattr(step, "step_type", ""):
            step_tools = self._step_types.tools_for(step.step_type)

        # Split the step's declaration: capability types compose the profile
        # (static ∪ dynamic − exclude), literal tool names pass through for
        # backward compatibility with old string lists.
        static_caps: list[type] = []
        for name in step_tools:
            if is_capability(name):
                static_caps.append(name)
            elif isinstance(name, str):
                add((name,))

        if static_caps or getattr(step, "grant_name", None):
            profile = ToolProfile(
                tools=tuple(static_caps),
                exclude=tuple(c for c in getattr(step, "exclude", ()) if is_capability(c)),
                grant_name=getattr(step, "grant_name", None),
            )
            add(tools_for_capabilities(evaluate_grant(profile, step)))

        return names

    def _resolve(self, name: str) -> Any:
        tool = self._resolve_optional(name)
        if tool is None:
            # Fail-fast at assembly — same shape as INVALID_SKILL.
            raise ValueError(f"INVALID_TOOL: {name}")
        return tool

    def _resolve_optional(self, name: str) -> Any | None:
        if name in self._internal:
            return self._internal[name]
        if name in self._external:
            return self._external[name]
        for tool in self._extra:
            if getattr(tool, "name", "") == name:
                return tool
        for tool in self._mcp_tools:
            if getattr(tool, "name", "") == name:
                return tool
        return None
