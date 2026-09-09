"""Unit tests for Pipeline/Step/Context Controller wiring (Phase 7)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from quro.core.catalog_adapter import StepCatalogAdapter
from quro.core.conformance import ConformanceChecker, SessionDependencies
from quro.core.cross_round_handoff import CrossRoundHandoff
from quro.core.domain.step_type import StepType, StepTypeCatalog
from quro.core.protocols import (
    AuditEvent,
    DependencyCheck,
    OperatorSummary,
    StepBuildResult,
    Violation,
)
from quro.core.round_controller import PlanGate, RoundController
from quro.core.round_scope import RoundScope
from quro.core.step_builder import StepBuilder
from quro.core.testing import InMemoryAuditLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_step_types() -> StepTypeCatalog:
    return StepTypeCatalog([
        StepType(name="specify", skill_pool=["research", "write"]),
        StepType(name="implement", skill_pool=["code", "test"]),
        StepType(name="evaluate", skill_pool=["review"], access="hints"),
    ])


def _make_builder(
    step_types: StepTypeCatalog | None = None,
    handoff: CrossRoundHandoff | None = None,
    audit_log: Any | None = None,
) -> StepBuilder:
    return StepBuilder(
        step_types=step_types or _make_step_types(),
        handoff=handoff,
        audit_log=audit_log,
    )


# ---------------------------------------------------------------------------
# StepBuilder tests
# ---------------------------------------------------------------------------


class TestStepBuilder:
    def test_create_eager_resolution(self):
        builder = _make_builder()
        result = builder.create(
            step_id="s1",
            objective="Research modules",
            step_type="specify",
            skills=["research"],
        )
        assert isinstance(result, StepBuildResult)
        assert result.dependency_check.ok
        assert result.spec.id == "s1"
        assert result.resolved_features == ()
        assert result.resolved_access == ""

    def test_dependency_same_round(self):
        builder = _make_builder()
        r1 = builder.create(step_id="s1", objective="First")
        assert r1.dependency_check.ok

        r2 = builder.create(
            step_id="s2",
            objective="Second",
            depends_on=["s1"],
        )
        assert r2.dependency_check.ok

    def test_dependency_cross_round(self):
        handoff = CrossRoundHandoff()
        builder = _make_builder(handoff=handoff)

        # Simulate a prior-round step.
        from quro.steps.core import StepSpec

        prior_spec = StepSpec(id="prior_step", objective="prior")
        prior_result = StepBuildResult(
            spec=prior_spec,
            resolved_features=(),
            resolved_tools=(),
            resolved_access="",
            dependency_check=DependencyCheck(ok=True),
        )
        handoff.carry_step_ref(0, "prior_step", prior_result)

        result = builder.create(
            step_id="s1",
            objective="Depends on prior",
            depends_on=["r0:prior_step"],
        )
        assert result.dependency_check.ok

    def test_cross_round_ref_not_found(self):
        builder = _make_builder(handoff=CrossRoundHandoff())
        result = builder.create(
            step_id="s1",
            objective="Bad ref",
            depends_on=["r0:nonexistent"],
        )
        assert not result.dependency_check.ok
        assert "not found" in result.dependency_check.errors[0]

    def test_skill_pool_rejection(self):
        builder = _make_builder()
        result = builder.create(
            step_id="s1",
            objective="Bad skills",
            step_type="specify",
            skills=["code"],  # not in specify's pool
        )
        assert not result.dependency_check.ok
        assert "not in pool" in result.dependency_check.errors[0]

    def test_duplicate_step_id(self):
        builder = _make_builder()
        builder.create(step_id="s1", objective="First")
        result = builder.create(step_id="s1", objective="Duplicate")
        assert not result.dependency_check.ok
        assert "already exists" in result.dependency_check.errors[0]


# ---------------------------------------------------------------------------
# RoundScope tests
# ---------------------------------------------------------------------------


class TestRoundScope:
    def test_topological_sort(self):
        scope = RoundScope(round_idx=0)
        builder = _make_builder()

        r1 = builder.create(step_id="s1", objective="First")
        r2 = builder.create(step_id="s2", objective="Second", depends_on=["s1"])
        r3 = builder.create(step_id="s3", objective="Third", depends_on=["s2"])

        scope.add(r1)
        scope.add(r2)
        scope.add(r3)

        ordered = scope.topological_batch()
        assert len(ordered) == 3
        ids = [s.id for s in ordered]
        assert ids.index("s1") < ids.index("s2")
        assert ids.index("s2") < ids.index("s3")

    def test_unexecuted_warning(self):
        scope = RoundScope(round_idx=0)
        builder = _make_builder()
        r1 = builder.create(step_id="s1", objective="First")
        scope.add(r1)
        assert scope.unexecuted_count() == 1


# ---------------------------------------------------------------------------
# AuditLog tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_round_scoped(self):
        log = InMemoryAuditLog()
        log.emit(AuditEvent(round_idx=0, kind="round_open", payload={}, ts=time.time()))
        log.emit(AuditEvent(round_idx=1, kind="round_open", payload={}, ts=time.time()))
        log.emit(AuditEvent(round_idx=0, kind="step_built", payload={}, ts=time.time()))

        r0_events = log.events_for_round(0)
        assert len(r0_events) == 2
        r1_events = log.events_for_round(1)
        assert len(r1_events) == 1


# ---------------------------------------------------------------------------
# ConformanceChecker tests
# ---------------------------------------------------------------------------


class TestConformanceChecker:
    def test_in_memory_rejection(self):
        checker = ConformanceChecker(testing=False)
        violations = checker.check_wiring(SessionDependencies(
            builder=_make_builder(),
            store=type("InMemoryStore", (), {})(),
            audit_log=InMemoryAuditLog(),
            controller=type("Controller", (), {})(),
        ))
        assert any(v.kind == "in_memory_store" for v in violations)

    def test_implicit_dependency(self):
        from quro.steps.core import StepSpec

        checker = ConformanceChecker(step_types=_make_step_types())

        class FakeBatch:
            pending_steps = {
                "s1": StepBuildResult(
                    spec=StepSpec(id="s1", objective="build X"),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
                "s2": StepBuildResult(
                    spec=StepSpec(
                        id="s2",
                        objective="test X from s1",
                        depends_on=[],
                    ),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
            }

        violations = checker.check_round_batch(FakeBatch())
        assert any(v.kind == "implicit_dependency" for v in violations)

    def test_unreachable_artifact_ref(self):
        """A step that references a prior artifact by id without declaring
        depends_on trips the unreachable_artifact_ref check (Anti-Pattern I)."""
        from quro.steps.core import StepSpec

        checker = ConformanceChecker(step_types=_make_step_types())

        class FakeBatch:
            pending_steps = {
                "s1": StepBuildResult(
                    spec=StepSpec(id="s1", objective="survey modules"),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
                "s2": StepBuildResult(
                    spec=StepSpec(
                        id="s2",
                        objective="go deeper on art_ab12cd34",
                        depends_on=[],
                    ),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
            }

        violations = checker.check_round_batch(FakeBatch())
        assert any(v.kind == "unreachable_artifact_ref" for v in violations)

    def test_unreachable_artifact_ref_declares_dep_is_ok(self):
        """Referencing a prior artifact id WITH a depends_on is allowed (we
        can't map the id to its owner at planning time)."""
        from quro.steps.core import StepSpec

        checker = ConformanceChecker(step_types=_make_step_types())

        class FakeBatch:
            pending_steps = {
                "s1": StepBuildResult(
                    spec=StepSpec(id="s1", objective="survey modules"),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
                "s2": StepBuildResult(
                    spec=StepSpec(
                        id="s2",
                        objective="go deeper on art_ab12cd34",
                        depends_on=["s1"],
                    ),
                    resolved_features=(),
                    resolved_tools=(),
                    resolved_access="",
                    dependency_check=DependencyCheck(ok=True),
                ),
            }

        assert not any(
            v.kind == "unreachable_artifact_ref"
            for v in checker.check_round_batch(FakeBatch())
        )


# ---------------------------------------------------------------------------
# CrossRoundHandoff tests
# ---------------------------------------------------------------------------


class TestCrossRoundHandoff:
    def test_carry_and_resolve(self):
        handoff = CrossRoundHandoff()
        art = {"body": "test artifact"}
        handoff.carry_artifact(0, "art_1", art)
        assert handoff.resolve_artifact("art_1") == art
        assert handoff.resolve_artifact("art_nonexistent") is None

    def test_carry_step_ref(self):
        handoff = CrossRoundHandoff()
        from quro.steps.core import StepSpec

        result = StepBuildResult(
            spec=StepSpec(id="step_a", objective="test"),
            resolved_features=(),
            resolved_tools=(),
            resolved_access="",
            dependency_check=DependencyCheck(ok=True),
        )
        handoff.carry_step_ref(1, "step_a", result)
        ref = handoff.resolve_step_ref("r1:step_a")
        assert ref is not None
        assert ref.spec.id == "step_a"
        assert handoff.resolve_step_ref("r0:step_a") is None


# ---------------------------------------------------------------------------
# IStepCatalogView tests
# ---------------------------------------------------------------------------


class TestStepCatalogView:
    def test_describe_operators_includes_skill_pool(self):
        catalog = _make_step_types()
        adapter = StepCatalogAdapter(catalog)
        summaries = adapter.describe_operators()
        assert len(summaries) == 3
        names = {s.name for s in summaries}
        assert names == {"specify", "implement", "evaluate"}

        specify = next(s for s in summaries if s.name == "specify")
        assert "research" in specify.skill_pool
        assert "write" in specify.skill_pool

    def test_operator_summary_fields(self):
        catalog = _make_step_types()
        summaries = catalog.describe_operators()
        for s in summaries:
            assert isinstance(s, OperatorSummary)
            assert s.name
            assert s.description
            assert s.cost == 1.0
            assert isinstance(s.skill_pool, tuple)
