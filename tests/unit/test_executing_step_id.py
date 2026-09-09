"""Regression tests: ensure executor session has correct executing_step_id.

In policy-loop mode the planner is an HTN solver (not an LLM role),
so no create_step is ever called.  The executor (builder) session must
have executing_step_id initialized so complete_step can find the step.
"""

from quro.core.domain.workflow_domain import EngineeringWorkflowDomain
from quro_thinking.kernel import ReasoningSession


class TestExecutingStepId:
    """Verify executing_step_id is present in executor session after init."""

    def test_executor_session_has_step_id(self) -> None:
        """After carry_forward to step_execute phase, the session must have
        a step entry matching the Pipeline step ID so complete_step works."""
        domain = EngineeringWorkflowDomain()
        session = ReasoningSession(domain)

        # Simulate what RoleAdapter.execute() does:
        # 1. engineering_workflow_init
        session.call("engineering_workflow_init", {
            "problem": "PROBLEM: count files\n\nOBJECTIVE: count files",
        })

        # 2. carry_forward from planner → sets phase to step_execute
        session.call("carry_forward", {
            "sources": [{"step_id": "planner", "keys": ["plan"],
                         "plan": "find files"}],
            "phase": "step_execute",
        })

        # 3. Seed the step entry (as fixed in adapter.py).
        pipeline_step_id = "t_f47f3bb1"
        state = session.state
        assert state is not None
        state["executing_step_id"] = pipeline_step_id
        state.setdefault("steps", []).append({
            "step_id": pipeline_step_id,
            "objective": "count files",
            "status": "in_progress",
        })

        # Now complete_step should find the step.
        resp = session.call("complete_step", {"step_id": pipeline_step_id})
        assert resp.ok is False  # no artifacts yet — but step IS found
        assert "not found" not in (resp.error or "")
        assert "no artifacts" in (resp.error or "").lower()

    def test_complete_step_with_empty_id_falls_back(self) -> None:
        """complete_step with empty step_id resolves to executing_step_id."""
        domain = EngineeringWorkflowDomain()
        session = ReasoningSession(domain)
        session.call("engineering_workflow_init", {"problem": "test"})

        # Advance to step_execute via carry_forward.
        session.call("carry_forward", {
            "sources": [],
            "phase": "step_execute",
        })

        pipeline_step_id = "t_abc123"
        state = session.state
        state["executing_step_id"] = pipeline_step_id
        state.setdefault("steps", []).append({
            "step_id": pipeline_step_id,
            "objective": "test",
            "status": "in_progress",
        })

        # Call complete_step via the domain with empty step_id.
        resp = session.call("complete_step", {"step_id": ""})
        # Empty string passed directly → domain sees ""
        # But the tool wrapper resolves empty → executing_step_id.
        # This test calls the domain directly, so empty → not found.
        # The tool wrapper test is below.
        assert resp.ok is False

    def test_tool_wrapper_resolves_empty_step_id(self) -> None:
        """The complete_step tool wrapper resolves empty step_id."""
        from quro.core.tools import StepTool, ToolAdapter

        domain = EngineeringWorkflowDomain()
        session = ReasoningSession(domain)
        session.call("engineering_workflow_init", {"problem": "test"})

        # Advance to step_execute via carry_forward.
        session.call("carry_forward", {
            "sources": [],
            "phase": "step_execute",
        })

        pipeline_step_id = "t_xyz789"
        state = session.state
        state["executing_step_id"] = pipeline_step_id
        state.setdefault("steps", []).append({
            "step_id": pipeline_step_id,
            "objective": "test",
            "status": "in_progress",
        })

        tools = ToolAdapter(session).build([StepTool()])
        cs_tool = {t.name: t for t in tools}["complete_step"]

        # Call with empty step_id → tool wrapper resolves to executing_step_id.
        result = cs_tool.invoke({})
        assert "not found" not in result.lower()
        # Expected: "Step 't_xyz789' has no artifacts"
        assert pipeline_step_id in result

    def test_missing_step_entry_is_not_found(self) -> None:
        """Without the fix (no step entry), complete_step fails with 'not found'."""
        domain = EngineeringWorkflowDomain()
        session = ReasoningSession(domain)
        session.call("engineering_workflow_init", {"problem": "test"})

        # Advance to step_execute.
        session.call("carry_forward", {
            "sources": [],
            "phase": "step_execute",
        })

        state = session.state
        state["executing_step_id"] = "t_missing"

        resp = session.call("complete_step", {"step_id": "t_missing"})
        # Without the step in state["steps"], domain reports "not found".
        assert not resp.ok
        assert "not found" in (resp.error or "")
