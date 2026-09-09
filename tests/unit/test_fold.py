from __future__ import annotations

from quro.core.domain import EngineeringWorkflowDomain
from quro.core.fold import handle_fold_tree, handle_terminate_subtree
from quro_thinking.kernel import ReasoningSession


def _make_session() -> ReasoningSession:
    domain = EngineeringWorkflowDomain()
    session = ReasoningSession(domain)
    session.call("engineering_workflow_init", {"problem": "test problem"})
    session.call("set_interpretation", {"text": "analysis"})
    session.call("set_plan", {"plan": "plan"})
    session.call("confirm_understanding", {})
    session.call("create_step", {"step_id": "s1", "objective": "step 1"})
    session.call("create_step", {"step_id": "s2", "objective": "step 2"})
    session.call("finalize_step", {})
    # Enter subtree phase so fold_tree is allowed
    session._state["phase"] = "step_execute:subtree"
    return session


def test_fold_tree_basic():
    session = _make_session()
    result = handle_fold_tree(
        session,
        subtree_id="s1",
        conclusions=[{"statement": "analysis complete", "confidence": 0.9}],
    )
    assert result["status"] == "folded"
    assert len(result["axioms"]) >= 1
    assert result["axioms"][0]["statement"] == "analysis complete"


def test_fold_tree_with_provided_axioms():
    session = _make_session()
    axioms = [
        {"statement": "pre-compressed fact", "justification_brief": "test"}
    ]
    result = handle_fold_tree(
        session,
        subtree_id="s1",
        conclusions=[],
        axioms=axioms,
    )
    assert result["status"] == "folded"
    assert len(result["axioms"]) == 1
    assert result["axioms"][0]["statement"] == "pre-compressed fact"


def test_fold_tree_multiple_conclusions():
    session = _make_session()
    conclusions = [
        {"statement": "fact 1", "confidence": 0.8},
        {"statement": "fact 2", "confidence": 0.95},
        {"statement": "fact 3", "confidence": 0.7},
    ]
    result = handle_fold_tree(session, subtree_id="s1", conclusions=conclusions)
    assert result["status"] == "folded"
    assert len(result["axioms"]) == 3


def test_terminate_subtree():
    session = _make_session()
    result = handle_terminate_subtree(
        session,
        reason="all approaches exhausted",
        derive_axiom="algorithm X is O(n2) in worst case",
    )
    assert result["status"] == "subtree_terminated"


def test_terminate_subtree_without_axiom():
    session = _make_session()
    result = handle_terminate_subtree(
        session,
        reason="no solution found",
    )
    assert result["status"] == "subtree_terminated"


def test_fold_updates_step_status():
    session = _make_session()
    handle_fold_tree(session, subtree_id="s1", conclusions=[])
    step = next(s for s in session.state["steps"] if s["step_id"] == "s1")
    assert step["round_status"] == "folded"
