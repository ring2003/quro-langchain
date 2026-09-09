"""Tests for StepFactory / PipelineFactory (phase 5 — the rebuild direction)."""

from __future__ import annotations

import pytest

from quro.core.domain.step_config import StepConfig
from quro.core.domain.step_factory import (
    IPipelineFactory,
    IStepFactory,
    PipelineFactory,
    RebuiltStep,
    StepFactory,
)
from quro.core.domain.step_type import StepType
from quro.core.features import Feature
from quro.steps.builtin import AuditLogHook
from quro.steps.core import StepSpec
from quro.steps.hooks import HookFactoryContext, HookRegistry


def _step_type() -> StepType:
    return StepType(
        "survey_module",
        skill_pool=["codegraph", "read-strategy"],
        identity_block="identity survey",
        executor_hints={"phase": "survey"},
        features=(Feature.RECOVERY.value,),
    )


def _spec() -> StepSpec:
    return StepSpec(
        id="s1",
        name="survey step",
        step_type="survey_module",
        objective="map the repo",
        expected_output="module list",
        validation="has artifacts",
        checklist=[{"item_id": "c1", "description": "read", "done": False}],
        tools=["commit_clue"],
        depends_on=["p1"],
        skills=["codegraph"],
        features=(Feature.RECOVERY.value,),
        message_strategy="accumulate",
        timeout=30,
        inputs=["p1"],
        hil=False,
    )


def _registry() -> HookRegistry:
    reg = HookRegistry()
    reg.register("audit_log", lambda ctx: AuditLogHook())
    return reg


def test_round_trip_rebuilds_spec_without_hooks():
    spec = _spec()
    cfg = StepConfig.from_step(spec, _step_type())
    rebuilt = StepFactory(_registry()).rebuild(cfg, hook_context=HookFactoryContext())
    assert isinstance(rebuilt, RebuiltStep)
    assert rebuilt.spec == spec  # dataclass field equality (no hooks → identical)
    assert rebuilt.spec.sub_steps == []


def test_rebuild_restores_step_type_inlined_fields():
    cfg = StepConfig.from_step(_spec(), _step_type())
    rebuilt = StepFactory(_registry()).rebuild(cfg, hook_context=HookFactoryContext())
    assert rebuilt.step_type.name == "survey_module"
    assert rebuilt.step_type.skill_pool == ["codegraph", "read-strategy"]
    assert rebuilt.step_type.identity_block == "identity survey"
    assert rebuilt.step_type.executor_hints == {"phase": "survey"}
    assert rebuilt.step_type.features == (Feature.RECOVERY.value,)


def test_rebuild_resolves_hook_names_through_registry():
    spec = _spec()
    spec.pre_hooks.append(AuditLogHook())
    spec.post_hooks.append(AuditLogHook())
    cfg = StepConfig.from_step(spec, _step_type())
    assert cfg.pre_hooks == ["audit_log"]
    assert cfg.post_hooks == ["audit_log"]

    rebuilt = StepFactory(_registry()).rebuild(cfg, hook_context=HookFactoryContext())
    assert [h.name for h in rebuilt.spec.pre_hooks] == ["audit_log"]
    assert [h.name for h in rebuilt.spec.post_hooks] == ["audit_log"]
    assert isinstance(rebuilt.spec.pre_hooks[0], AuditLogHook)


def test_rebuild_unknown_hook_raises():
    spec = _spec()
    spec.pre_hooks.append(AuditLogHook())
    cfg = StepConfig.from_step(spec, _step_type())
    with pytest.raises(KeyError, match="Unknown hook"):
        StepFactory(HookRegistry()).rebuild(cfg, hook_context=HookFactoryContext())


def test_rebuild_sub_steps_recursively():
    inner = StepSpec(id="inner", objective="leaf")
    outer = _spec()
    outer.sub_steps.append(inner)
    cfg = StepConfig.from_step(outer, _step_type())
    rebuilt = StepFactory(_registry()).rebuild(cfg, hook_context=HookFactoryContext())
    assert len(rebuilt.spec.sub_steps) == 1
    assert rebuilt.spec.sub_steps[0] == inner


def test_factory_satisfies_protocol():
    assert isinstance(StepFactory(_registry()), IStepFactory)
    assert isinstance(PipelineFactory(_registry()), IPipelineFactory)


def test_pipeline_factory_topological_order():
    a = StepConfig(id="a")
    b = StepConfig(id="b", depends_on=["a"])
    c = StepConfig(id="c", depends_on=["a", "b"])
    configs = [c, a, b]  # intentionally out of order
    rebuilt = PipelineFactory(_registry()).rebuild(
        configs, hook_context=HookFactoryContext()
    )
    assert [r.spec.id for r in rebuilt] == ["a", "b", "c"]


def test_pipeline_factory_cycle_raises():
    a = StepConfig(id="a", depends_on=["b"])
    b = StepConfig(id="b", depends_on=["a"])
    with pytest.raises(ValueError, match="cycle"):
        PipelineFactory(_registry()).rebuild(
            [a, b], hook_context=HookFactoryContext()
        )


def test_pipeline_factory_unknown_dep_raises():
    a = StepConfig(id="a", depends_on=["missing"])
    with pytest.raises(ValueError, match="unknown step"):
        PipelineFactory(_registry()).rebuild([a], hook_context=HookFactoryContext())
