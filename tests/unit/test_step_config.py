"""Tests for StepConfig (phase 4 — the config half of a checkpoint)."""

from __future__ import annotations

import json

from quro.core.domain.step_config import StepConfig
from quro.core.domain.step_type import StepType
from quro.core.features import Feature
from quro.steps.builtin import AuditLogHook
from quro.steps.core import StepSpec


def _step_type() -> StepType:
    return StepType(
        "survey_module",
        skill_pool=["codegraph", "read-strategy"],
        identity_block="identity survey",
        executor_hints={"phase": "survey", "terminal_tool": "complete_step"},
        features=(Feature.RECOVERY,),
    )


def _step() -> StepSpec:
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
        hil=False,
    )


def test_from_step_inlines_identity_and_features():
    cfg = StepConfig.from_step(_step(), _step_type())
    assert cfg.id == "s1"
    assert cfg.step_type == "survey_module"
    assert cfg.identity_block == "identity survey"
    assert cfg.executor_hints["phase"] == "survey"
    assert cfg.features == ("recovery",)
    assert cfg.skills == ["codegraph"]
    assert cfg.message_strategy == "accumulate"
    assert cfg.tools == ["commit_clue"]


def test_from_step_without_step_type_uses_empty_inlines():
    cfg = StepConfig.from_step(_step(), None)
    assert cfg.identity_block == ""
    assert cfg.executor_hints == {}
    assert cfg.features == ("recovery",)  # falls back to the StepSpec mirror


def test_round_trips_json():
    cfg = StepConfig.from_step(_step(), _step_type())
    data = json.loads(json.dumps(cfg.to_dict()))
    restored = StepConfig.from_dict(data)
    assert restored == cfg
    assert restored.features == ("recovery",)


def test_from_dict_defaults():
    cfg = StepConfig.from_dict({"id": "x"})
    assert cfg.id == "x"
    assert cfg.features == ()
    assert cfg.depth == 0
    assert cfg.parent_path == ""
    assert cfg.hil is False


# ---------------------------------------------------------------------------
# phase 5 — v0.2 fields (decision 32)
# ---------------------------------------------------------------------------


def test_from_step_inlines_skill_pool_hooks_timeout_inputs():
    step = StepSpec(
        id="s2",
        step_type="survey_module",
        objective="map the repo",
        skills=["codegraph"],
        features=(Feature.RECOVERY.value,),
        timeout=30,
        inputs=["p1"],
        pre_hooks=[AuditLogHook()],
        post_hooks=[AuditLogHook()],
    )
    cfg = StepConfig.from_step(step, _step_type())
    assert cfg.skill_pool == ["codegraph", "read-strategy"]
    assert cfg.pre_hooks == ["audit_log"]
    assert cfg.post_hooks == ["audit_log"]
    assert cfg.timeout == 30
    assert cfg.inputs == ["p1"]


def test_sub_steps_recurse_with_parent_path_and_depth():
    inner = StepSpec(id="inner", objective="leaf")
    outer = StepSpec(id="outer", objective="root", sub_steps=[inner])
    cfg = StepConfig.from_step(outer)
    assert cfg.depth == 0
    assert cfg.parent_path == ""
    assert len(cfg.sub_steps) == 1
    sub = cfg.sub_steps[0]
    assert sub.id == "inner"
    assert sub.depth == 1
    assert sub.parent_path == "outer"
    assert sub.sub_steps == []


def test_round_trips_json_with_v02_fields():
    inner = StepSpec(id="inner", objective="leaf")
    step = StepSpec(
        id="s3",
        step_type="survey_module",
        objective="map the repo",
        skills=["codegraph"],
        timeout=7,
        inputs=["p1"],
        pre_hooks=[AuditLogHook()],
        sub_steps=[inner],
    )
    cfg = StepConfig.from_step(step, _step_type())
    data = json.loads(json.dumps(cfg.to_dict()))
    restored = StepConfig.from_dict(data)
    assert restored == cfg
    assert restored.skill_pool == ["codegraph", "read-strategy"]
    assert restored.pre_hooks == ["audit_log"]
    assert restored.timeout == 7
    assert restored.sub_steps[0].id == "inner"
