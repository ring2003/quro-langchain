"""Foundational protocols and data types for the Pipeline/Step/Context Controller.

All protocol ABCs and dataclasses live here. This module has zero runtime
dependencies — implementations import protocols, never the reverse.

Phase 0 deliverable (architecture §3.1–§3.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


# --- StepBuildResult (architecture §3.1) ---


@dataclass(frozen=True)
class DependencyCheck:
    """Structural dependency validation result."""

    ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepBuildResult:
    """Self-describing step construction product -- replaces bare StepSpec
    as the tool-visible product."""

    spec: Any  # StepSpec -- lazy import to avoid circular dep
    resolved_features: tuple[str, ...]
    resolved_tools: tuple[Any, ...]
    resolved_access: str
    dependency_check: DependencyCheck
    warnings: tuple[str, ...] = ()


# --- IStepBuilder (architecture §3.1) ---


@runtime_checkable
class IStepBuilder(Protocol):
    def create(
        self,
        step_id: str,
        objective: str,
        step_type: str = "",
        depends_on: Sequence[str] = (),
        skills: Sequence[str] = (),
        expected_output: str = "",
        validation: str = "",
    ) -> StepBuildResult: ...

    def update(self, step_id: str, **fields: Any) -> StepBuildResult: ...

    def list_pending(self) -> list[StepBuildResult]: ...


# --- ICrossRoundHandoff (architecture §3.2) ---


@runtime_checkable
class ICrossRoundHandoff(Protocol):
    def carry_artifact(self, round_idx: int, artifact_id: str, artifact: Any) -> None: ...
    def resolve_artifact(self, artifact_id: str) -> Any | None: ...
    def carry_step_ref(self, round_idx: int, step_id: str, result: StepBuildResult) -> None: ...
    def resolve_step_ref(self, qualified_id: str) -> StepBuildResult | None: ...


# --- IPipelineController (architecture §3.3) ---


@runtime_checkable
class IPipelineController(Protocol):
    def execute(self, batch: Any) -> Any: ...  # batch: RoundScope, returns ExecutionResult


# --- IStepCatalogView (architecture §3.4) ---


@dataclass(frozen=True)
class OperatorSummary:
    name: str
    description: str
    cost: float
    skill_pool: tuple[str, ...]
    access: str


@runtime_checkable
class IStepCatalogView(Protocol):
    def describe_operators(self) -> list[OperatorSummary]: ...


# --- IAuditLog (architecture §3.5) ---


@dataclass(frozen=True)
class AuditEvent:
    round_idx: int
    kind: str  # "round_open" | "step_built" | "dependency_rejected" |
               # "pipeline_executed" | "plan_overwritten" |
               # "gate_closed" | "conformance_violation" |
               # "step_carried_unexecuted" | ...
    payload: dict[str, Any]
    ts: float


@runtime_checkable
class IAuditLog(Protocol):
    def emit(self, event: AuditEvent) -> None: ...
    def events_for_round(self, round_idx: int) -> list[AuditEvent]: ...


# --- IConformanceChecker (architecture §3.7) ---


@dataclass(frozen=True)
class Violation:
    kind: str  # "in_memory_store" | "missing_protocol" | "implicit_dependency" | ...
    component: str
    message: str


@runtime_checkable
class IConformanceChecker(Protocol):
    def check_wiring(self, session_deps: Any) -> list[Violation]: ...
    def check_round_batch(self, batch: Any) -> list[Violation]: ...
