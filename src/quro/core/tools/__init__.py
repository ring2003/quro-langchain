"""Declarative tool framework — resource-scoped toolsets.

Tools are organised by resource domain (toolset) and mapped to runtime tools by
a single ``ToolAdapter``.  A tool's name, schema, description and handler all
come from one source (the toolset method).  Allocation (who gets which tool) is
the ``ToolCoordinator``'s job, not a declaration on the toolset.
"""

from __future__ import annotations

from quro.core.tools.spec import (
    RESOURCE_KIND_ARTIFACT,
    RESOURCE_KIND_CLUE,
    RESOURCE_KIND_PLAN,
    RESOURCE_KIND_SESSION,
    RESOURCE_KIND_STEP,
    ToolAdapter,
    ToolSpec,
    Toolset,
    format_response,
)
from quro.core.tools.coordinator import (
    ARTIFACT_TOOLS,
    CONTRACT_TOOLS,
    EVALUATOR_TOOLS,
    PLANNER_TOOLS,
    PRINCIPAL_EVALUATOR,
    PRINCIPAL_PLANNER,
    PRINCIPAL_WORKER,
    RECOVERY_TOOLS,
    IToolCoordinator,
    ToolCoordinator,
)
from quro.core.tools.sets import (
    ArtifactTool,
    ClarificationTool,
    EvaluationTool,
    PlanningTool,
    RecoveryTool,
    ReportTool,
    SessionTool,
    StepTool,
    domain_toolsets,
    framework_toolsets,
)

__all__ = [
    "RESOURCE_KIND_ARTIFACT",
    "RESOURCE_KIND_CLUE",
    "RESOURCE_KIND_PLAN",
    "RESOURCE_KIND_SESSION",
    "RESOURCE_KIND_STEP",
    "ToolAdapter",
    "ToolSpec",
    "Toolset",
    "format_response",
    "ARTIFACT_TOOLS",
    "CONTRACT_TOOLS",
    "EVALUATOR_TOOLS",
    "PLANNER_TOOLS",
    "RECOVERY_TOOLS",
    "PRINCIPAL_WORKER",
    "PRINCIPAL_PLANNER",
    "PRINCIPAL_EVALUATOR",
    "IToolCoordinator",
    "ToolCoordinator",
    "StepTool",
    "ArtifactTool",
    "PlanningTool",
    "EvaluationTool",
    "ClarificationTool",
    "SessionTool",
    "ReportTool",
    "RecoveryTool",
    "framework_toolsets",
    "domain_toolsets",
]
