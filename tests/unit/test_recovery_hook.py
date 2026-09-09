"""Tests for Phase D — the recovery hook (interruption salvage)."""

from __future__ import annotations

from quro.core.features import Feature
from quro.steps.core import StepResult, StepSpec
from quro.steps.hooks import HookContext
from quro.steps.recovery_hook import (
    RecoveryHook,
    _build_recovery_prompt,
    _step_artifacts,
    attach_recovery_hooks,
    should_recover,
)


def _spec(step_type="survey_module") -> StepSpec:
    features = (Feature.RECOVERY.value,) if step_type == "survey_module" else ()
    return StepSpec(
        id="s1", step_type=step_type, objective="map the path", features=features,
    )


def _result_with_artifact() -> StepResult:
    return StepResult.success("s1", state={"artifacts": [{"step_id": "s1"}]})

def _result_empty(clues=None) -> StepResult:
    recovery = {"s1": {
        "step_round": 0,
        "clues": clues or [{"text": "c1"}],
        "continue_reasons": [],
        "failed_reasons": [],
    }}
    return StepResult.failure("s1", "no artifact", state={"artifacts": [], "recovery": recovery})


def test_should_recover():
    spec = _spec()
    assert should_recover(spec, _result_with_artifact()) is False
    assert should_recover(spec, _result_empty()) is True
    # not recovery-enabled step type
    assert should_recover(_spec("assemble_report"), _result_empty()) is False


def test_step_artifacts_scoped_to_step():
    state = {"artifacts": [
        {"step_id": "s1"},
        {"step_id": "s2"},
    ]}
    result = StepResult.success("s1", state=state)
    assert [a["step_id"] for a in _step_artifacts("s1", result)] == ["s1"]
    assert _step_artifacts("s3", result) == []


def test_build_recovery_prompt_contains_journal_and_decision():
    spec = _spec()
    prompt = _build_recovery_prompt(
        spec, 3,
        [{"text": "c1"}],
        {"continue_reasons": ["need callers"], "failed_reasons": []},
        3,
    )
    assert "STEP RECOVERY" in prompt
    assert "c1" in prompt
    assert "need callers" in prompt
    assert "complete_step" in prompt
    assert "continue_step" in prompt
    assert "give up" in prompt
    assert "last recovery round" in prompt  # round == max


def test_attach_recovery_hooks_only_for_recovery_enabled():
    steps = [
        StepSpec(id="a", step_type="survey_module", features=(Feature.RECOVERY.value,)),
        StepSpec(id="b", step_type="assemble_report"),
    ]
    attach_recovery_hooks(steps, object())
    assert len(steps[0].post_hooks) == 1
    assert steps[0].post_hooks[0].name == "recovery"
    assert steps[1].post_hooks == []


def test_recovery_hook_returns_none_when_step_succeeded():
    hook = RecoveryHook(object())
    ctx = HookContext(problem="p")
    result = hook.on_post_step(_spec(), _result_with_artifact(), ctx)
    assert result is None


def test_recovery_hook_returns_none_for_wrong_step_type():
    hook = RecoveryHook(object())
    ctx = HookContext(problem="p")
    result = hook.on_post_step(_spec("assemble_report"), _result_empty(), ctx)
    assert result is None


def test_recovery_hook_no_clues_returns_last_result():
    """With no clues to resume from, the hook returns the failed result (policy
    loop backstops) rather than re-running."""
    hook = RecoveryHook(object())
    ctx = HookContext(problem="p")
    empty = _result_empty(clues=[])
    result = hook.on_post_step(_spec(), empty, ctx)
    assert result is not None
    assert result.ok is False
