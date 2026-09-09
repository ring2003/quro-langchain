"""Tests for ``src/quro/algorithm/`` — pure functions, zero dependencies.

These tests should run without any quro-langchain setup.
"""

from quro.algorithm.goal_status import (
    Disposition,
    UNSATLevel,
    classify_unsat,
    compute_goal_status,
)


class TestComputeGoalStatus:
    """Test the pure compute_goal_status with plain dict input."""

    def test_all_sat(self) -> None:
        sat, unsat, unknown = compute_goal_status(
            {"a": True, "b": True}, ["a", "b"],
        )
        assert sat == ["a", "b"]
        assert unsat == []
        assert unknown == []

    def test_some_unsat(self) -> None:
        sat, unsat, unknown = compute_goal_status(
            {"a": True, "b": False}, ["a", "b", "c"],
        )
        assert sat == ["a"]
        assert set(unsat) == {"b", "c"}
        assert unknown == ["c"]

    def test_none_is_unknown_and_unsat(self) -> None:
        sat, unsat, unknown = compute_goal_status(
            {"a": None}, ["a"],
        )
        assert sat == []
        assert unsat == ["a"]
        assert unknown == ["a"]

    def test_missing_is_unknown_and_unsat(self) -> None:
        sat, unsat, unknown = compute_goal_status({}, ["a"])
        assert unsat == ["a"]
        assert unknown == ["a"]

    def test_empty_goals(self) -> None:
        sat, unsat, unknown = compute_goal_status({"a": True}, [])
        assert sat == []
        assert unsat == []
        assert unknown == []

    def test_truthy_non_bool_is_not_sat(self) -> None:
        """Only True counts as SAT. 1 is not True."""
        sat, unsat, unknown = compute_goal_status(
            {"a": True, "b": 1}, ["a", "b"],
        )
        assert sat == ["a"]
        assert "b" in unsat


class TestClassifyUnsat:
    """Test the pure classify_unsat decision function."""

    def test_sat(self) -> None:
        raw = classify_unsat(unsat_facts=[], sat_facts=["a", "b"])
        assert raw["level"] == UNSATLevel.VERIFY_UNSAT
        assert raw["disposition"] == Disposition.CONTINUE
        assert raw["unsat_facts"] == []

    def test_proof_unsat(self) -> None:
        raw = classify_unsat(proof_type="resource_exhaustion")
        assert raw["level"] == UNSATLevel.PROOF_UNSAT
        assert raw["disposition"] == Disposition.ABORT

    def test_plan_unsat_first_round(self) -> None:
        raw = classify_unsat(pef_empty=True, round_index=0)
        assert raw["level"] == UNSATLevel.PLAN_UNSAT
        assert raw["disposition"] == Disposition.CONTINUE

    def test_plan_unsat_after_retry(self) -> None:
        raw = classify_unsat(pef_empty=True, round_index=2)
        assert raw["level"] == UNSATLevel.PLAN_UNSAT
        assert raw["disposition"] == Disposition.ESCALATE_L1

    def test_run_unsat(self) -> None:
        raw = classify_unsat(step_error="crash", step_id="s1")
        assert raw["level"] == UNSATLevel.RUN_UNSAT
        assert raw["disposition"] == Disposition.RETRY_STEP

    def test_verify_unsat_fresh(self) -> None:
        raw = classify_unsat(
            unsat_facts=["a"],
            previous_unsat_facts=["b"],  # different
        )
        assert raw["level"] == UNSATLevel.VERIFY_UNSAT
        assert raw["disposition"] == Disposition.ADD_STEPS
        assert raw["stall_count"] == 0

    def test_verify_unsat_stalled(self) -> None:
        raw = classify_unsat(
            unsat_facts=["a"],
            previous_unsat_facts=["a"],  # same
            max_stall_rounds=1,
        )
        assert raw["level"] == UNSATLevel.VERIFY_UNSAT
        assert raw["disposition"] == Disposition.ESCALATE_L1
        assert raw["stall_count"] == 1

    def test_proof_priority_over_plan(self) -> None:
        raw = classify_unsat(pef_empty=True, proof_type="state_trap")
        assert raw["level"] == UNSATLevel.PROOF_UNSAT

    def test_plan_priority_over_run(self) -> None:
        raw = classify_unsat(pef_empty=True, step_error="crash")
        assert raw["level"] == UNSATLevel.PLAN_UNSAT

    def test_output_keys(self) -> None:
        """All expected keys are present."""
        raw = classify_unsat()
        expected = {
            "level", "disposition", "unsat_facts", "sat_facts",
            "cause", "step_id", "step_error", "proof_type", "stall_count",
        }
        assert expected <= set(raw.keys())
