"""Tests for HookContext, HookRegistry, and HIL step/tool.

The hook-lifecycle tests that constructed ``StepSpec(role=…)`` were removed:
the ``role`` concept was eliminated in Phase 0 (a step's identity is its
``StepType``), so those fixtures no longer exist.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quro.steps.hooks import HookContext, HookFactoryContext, HookRegistry
from quro.steps.builtin import AuditLogHook, PromptInjectHook
from quro.steps.hil_step import HILBackend, HumanInLoopStep
from quro.steps.hil_tool import make_hil_tool


# ---------------------------------------------------------------------------
# HookContext
# ---------------------------------------------------------------------------


class TestHookContext:
    def test_defaults(self):
        ctx = HookContext(problem="test")
        assert ctx.problem == "test"
        assert ctx.step_results == {}
        assert ctx.system_prompt is None
        assert ctx.extra_tools == []
        assert ctx.remove_tools == set()
        assert ctx.hil_pending is False

    def test_from_pipeline_context(self):
        ctx = HookContext.from_pipeline_context(
            "p", {"s1": MagicMock()}
        )
        assert ctx.problem == "p"
        assert "s1" in ctx.step_results


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


class TestHookRegistry:
    def test_register_and_create(self):
        reg = HookRegistry()
        reg.register("audit_log", lambda ctx: AuditLogHook())
        hook = reg.create("audit_log")
        assert isinstance(hook, AuditLogHook)

    def test_factory_receives_context(self):
        received: dict[str, object] = {}

        def factory(ctx: HookFactoryContext) -> AuditLogHook:
            received["executor"] = ctx.step_executor
            return AuditLogHook()

        reg = HookRegistry()
        reg.register("ctx_hook", factory)
        executor = object()
        hook = reg.create("ctx_hook", HookFactoryContext(step_executor=executor))
        assert isinstance(hook, AuditLogHook)
        assert received["executor"] is executor

    def test_create_defaults_to_empty_context(self):
        reg = HookRegistry()
        reg.register("audit_log", lambda ctx: AuditLogHook())
        assert isinstance(reg.create("audit_log"), AuditLogHook)

    def test_create_unknown_raises(self):
        reg = HookRegistry()
        with pytest.raises(KeyError, match="Unknown hook"):
            reg.create("nonexistent")

    def test_create_from_configs(self):
        reg = HookRegistry()
        reg.register("audit_log", lambda ctx: AuditLogHook())
        reg.register("prompt_inject", lambda ctx: PromptInjectHook())

        hooks = reg.create_from_configs([
            {"name": "audit_log"},
            {"name": "prompt_inject", "config": {"extra_instructions": "hi"}},
        ])
        assert len(hooks) == 2
        assert isinstance(hooks[0], AuditLogHook)
        assert isinstance(hooks[1], PromptInjectHook)
        assert hooks[1].extra_instructions == "hi"

    def test_list_names(self):
        reg = HookRegistry()
        reg.register("a", lambda ctx: AuditLogHook())
        reg.register("b", lambda ctx: AuditLogHook())
        assert reg.list_names() == ["a", "b"]

    def test_overwrite_warns(self, caplog):
        reg = HookRegistry()
        reg.register("x", lambda ctx: AuditLogHook())
        reg.register("x", lambda ctx: AuditLogHook())
        assert "overwriting" in caplog.text.lower()


# ---------------------------------------------------------------------------
# HumanInLoopStep
# ---------------------------------------------------------------------------


class TestHumanInLoopStep:
    def test_choices_default(self):
        hil = HumanInLoopStep(id="hil")
        assert hil.choices == []

    def test_render_prompt_returns_objective(self):
        hil = HumanInLoopStep(id="hil", objective="approve?")
        prompt = hil.render_prompt({})
        assert prompt == "approve?"

    def test_render_prompt_falls_back_if_no_jinja(self):
        hil = HumanInLoopStep(
            id="hil",
            objective="fallback",
            prompt_template="Hello {{ objective }}",
        )
        prompt = hil.render_prompt({})
        # jinja2 is available in test env, so template renders
        assert prompt == "Hello fallback"


# ---------------------------------------------------------------------------
# HILBackend protocol
# ---------------------------------------------------------------------------


class TestHILBackend:
    def test_protocol_check(self):
        class CLI:
            def ask_user(self, question, *, choices=None, timeout=0):
                return "yes"

        assert isinstance(CLI(), HILBackend)


# ---------------------------------------------------------------------------
# make_hil_tool
# ---------------------------------------------------------------------------


class TestMakeHILTool:
    def test_tool_creation(self):
        backend = MagicMock(spec=HILBackend)
        backend.ask_user.return_value = "yes"
        tool = make_hil_tool(backend)
        assert tool.name == "ask_user"
        result = tool.invoke({"question": "proceed?"})
        assert result == "yes"
        backend.ask_user.assert_called_once_with("proceed?", choices=None)

    def test_tool_with_choices(self):
        backend = MagicMock(spec=HILBackend)
        backend.ask_user.return_value = "approve"
        tool = make_hil_tool(backend)
        result = tool.invoke({"question": "select", "choices": "yes, no"})
        assert result == "approve"
        backend.ask_user.assert_called_once_with("select", choices=["yes", "no"])
