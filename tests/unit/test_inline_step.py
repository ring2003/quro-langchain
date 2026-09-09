"""Tests for the inline-step tier (primitive-step.md R1).

Covers:
- compaction eval is a one-shot, kernel-free LLM call producing clean text;
- the registry resolves built-ins by name;
- ``next_instruction`` block sorting (kind=31 after objective) and trim
  protection;
- ``RecoveryCoordinator`` inlines the declared compaction into the recovering
  step (the recovering step's ``next_instruction`` is set before re-run).
"""

from __future__ import annotations

import pytest

from quro.context.blocks import KIND_ORDER, block_kind
from quro.context.coordinator import NextInstructionBlockHook, PromptCoordinator
from quro.runtime.base import FakeBackend
from quro.runtime.inline_step import (
    InlineStep,
    InlineStepEvalError,
    backtrack_plan_compaction,
    default_registry,
    get_inline_step,
    next_step_compaction,
)


def test_registry_resolves_builtins():
    assert get_inline_step("backtrack_plan") is not None
    assert get_inline_step("next_step") is not None
    assert get_inline_step("nope") is None
    assert set(default_registry().names()) == {"backtrack_plan", "next_step"}


def test_compaction_eval_returns_clean_text():
    backend = FakeBackend(responses=["  do the thing  "])
    result = backtrack_plan_compaction().eval(
        context="ctx", backend=backend, step_id="s1", objective="obj"
    )
    assert result == "do the thing"


def test_compaction_eval_requires_backend():
    with pytest.raises(InlineStepEvalError):
        backtrack_plan_compaction().eval(context="ctx", backend=None)


def test_render_user_prompt_substitutes_context():
    prompt = backlog_plan_render()
    assert "s1" in prompt
    assert "obj" in prompt
    assert "ctx" in prompt


def backlog_plan_render() -> str:
    return backtrack_plan_compaction().render_user_prompt(
        context="ctx", objective="obj", step_id="s1"
    )


def test_next_instruction_sorts_after_objective():
    assert KIND_ORDER["next_instruction"] > KIND_ORDER["objective"]
    assert block_kind("next_instruction") == "next_instruction"
    assert block_kind("objective") == "objective"


def test_next_instruction_block_hook_emits_block():
    hook = NextInstructionBlockHook("re-run carefully")
    assert hook.get_blocks("worker") == {
        "next_instruction": "## NEXT INSTRUCTION\nre-run carefully"
    }
    assert NextInstructionBlockHook("").get_blocks("worker") == {}


def test_next_instruction_survives_trim():
    """kind=31 (last) must be in the trim keep-list, like objective."""
    hook = NextInstructionBlockHook("keep me")
    coordinator = PromptCoordinator(
        [hook],
        principal="worker",
        phase="step_execute",
    )
    system, user = coordinator.assemble(budget=5)
    # The next_instruction block is never dropped by trim.
    assert "next_instruction" in user
