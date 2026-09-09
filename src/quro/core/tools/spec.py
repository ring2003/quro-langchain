"""Declarative tool framework — resource-scoped toolsets.

A toolset declares semantic tools as plain public methods over a
``resource_kind``; a single ``ToolAdapter`` maps them to runtime tools.  Tools
are organised by *resource* (toolset) rather than as a flat list, so the
surface is ``O(toolsets)``.

The model::

    resource-kind(step/artifact/clue/plan/session) → tool

    StepTool(kind=step):        create_step / get_step / update_step / complete_step / ...
    ArtifactTool(kind=artifact): add_artifact / get_artifact
    ...

Each toolset subclasses :class:`Toolset`, declares ``resource_kind``, and
implements its tools as public methods.  The method name is the tool name, the
method signature is the tool's argument schema (single source of truth), the
docstring is the tool description, and the method body is the handler.  There
is no separate hand-written tool table to drift from the domain op.

Tool *allocation* (who gets which tool) is no longer declared on the toolset —
it is the ``ToolCoordinator``'s job (contract / lifecycle / ``StepType.tools``).
A tool's resource effect is a runtime fact (which primitive it calls), not a
declared verb/scope label.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from langchain_core.tools import StructuredTool


# ---------------------------------------------------------------------------
# Resource kinds (what the tool operates on)
# ---------------------------------------------------------------------------

# Phase-1 tool-declaration resource kinds (design doc §8): a toolset declares
# the resource it operates on.  ``session`` covers DomainState scalar/collection
# fields that are not addressable resources (checkpoint / evaluation / …).
RESOURCE_KIND_STEP = "step"
RESOURCE_KIND_ARTIFACT = "artifact"
RESOURCE_KIND_CLUE = "clue"
RESOURCE_KIND_PLAN = "plan"
RESOURCE_KIND_SESSION = "session"


# ---------------------------------------------------------------------------
# ToolSpec — a declared semantic tool
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Declared tool: name + resource_kind + schema + handler."""

    name: str
    resource_kind: str
    description: str
    handler: Callable[..., str]
    parameters: list[inspect.Parameter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Toolset — a resource-scoped set of semantic tools
# ---------------------------------------------------------------------------


class Toolset:
    """Base for resource-scoped tool sets.

    Subclasses declare ``resource_kind`` (one of the phase-1 resource kinds)
    and implement tools as public methods.  ``collect()`` inspects the public
    methods and returns their :class:`ToolSpec` declarations.
    """

    resource_kind: str = RESOURCE_KIND_SESSION

    def collect(self) -> list[ToolSpec]:
        """Return the tool specs declared by this toolset, in declaration order."""
        specs: list[ToolSpec] = []
        for name in dir(type(self)):
            if name.startswith("_") or name == "collect":
                continue
            fn = getattr(type(self), name, None)
            if fn is None or not callable(fn):
                continue
            bound = getattr(self, name)
            sig = inspect.signature(bound)
            params = [
                p for p in sig.parameters.values()
                if p.name not in ("self", "session")
            ]
            specs.append(ToolSpec(
                name=name,
                resource_kind=self.resource_kind,
                description=(fn.__doc__ or "").strip(),
                handler=bound,
                parameters=params,
            ))
        return specs


# ---------------------------------------------------------------------------
# Response formatting — ToolResponse → LLM-facing string
# ---------------------------------------------------------------------------


def format_response(name: str, resp: Any) -> str:
    """Render a kernel ``ToolResponse`` into the string a tool returns."""
    if not resp.ok:
        lines = [f"Error: {resp.error}"]
        if resp.explanation:
            lines.append(resp.explanation)
        return "\n".join(lines)
    if resp.explanation:
        return resp.explanation
    return f"{name} ok"


# ---------------------------------------------------------------------------
# ToolAdapter — IToolAdapter: toolset declarations → runtime tools
# ---------------------------------------------------------------------------


class ToolAdapter:
    """Adapt a set of :class:`Toolset` declarations into runtime tools.

    The adapter binds each declared tool to the given ``session`` and derives
    the tool's argument schema from the method signature — so a tool's name,
    description, schema and handler all come from one source (the toolset
    method).
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    def build(self, toolsets: Iterable[Toolset]) -> list[StructuredTool]:
        """Return runtime tools for every declared spec in *toolsets*."""
        tools: list[StructuredTool] = []
        for toolset in toolsets:
            for spec in toolset.collect():
                tools.append(self._to_langchain(spec))
        return tools

    def _to_langchain(self, spec: ToolSpec) -> StructuredTool:
        session = self._session

        def _invoke(**kwargs: Any) -> str:
            return spec.handler(session, **kwargs)

        # pydantic's validate_arguments reads both the signature and the
        # __annotations__ dict, so reconstruct both from the declared params.
        params: list[inspect.Parameter] = []
        annotations: dict[str, Any] = {"return": str}
        for p in spec.parameters:
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
            annotations[p.name] = ann
            params.append(p.replace(annotation=ann))
        _invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=str,
        )
        _invoke.__annotations__ = annotations  # type: ignore[attr-defined]
        _invoke.__doc__ = spec.description or spec.name
        tool = StructuredTool.from_function(
            func=_invoke,
            name=spec.name,
            description=spec.description or spec.name,
        )
        tool.metadata = {
            "resource_kind": spec.resource_kind,
        }
        return tool
