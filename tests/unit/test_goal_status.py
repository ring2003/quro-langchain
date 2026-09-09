"""Tests for ``src/quro/planner/goal_status.py`` — WorldState abstraction,
UNSAT classification, fact extraction, ProblemSignature, and RoundRecord.
"""

from __future__ import annotations

import pytest

from quro.pipeline.core import PipelineResult
from quro.planner.goal_status import (
    AuditorFactExtractor,
    Disposition,
    Fact,
    FactExtractor,
    FactSource,
    ProblemSignature,
    RoundRecord,
    UNSATDiagnosis,
    UNSATLevel,
    WorldState,
    classify_unsat,
    compute_goal_status,
    goal_status_from_results,
)
from quro.steps.core import StepResult


# ===========================================================================
# Fact
# ===========================================================================


class TestFact:
    def test_defaults(self) -> None:
        f = Fact(name="tests_pass", value=True)
        assert f.name == "tests_pass"
        assert f.value is True
        assert f.source == FactSource.INITIAL
        assert f.confidence == 1.0
        assert f.sequence == 0
        assert f.prior_value is None

    def test_is_true(self) -> None:
        assert Fact(name="a", value=True).is_true() is True
        assert Fact(name="b", value=False).is_true() is False
        assert Fact(name="c", value=1).is_true() is False

    def test_is_known(self) -> None:
        assert Fact(name="a", value=True).is_known() is True
        assert Fact(name="b", value=False).is_known() is True
        assert Fact(name="c", value=None).is_known() is False

    def test_roundtrip_dict(self) -> None:
        f = Fact(
            name="tests_pass",
            value=True,
            source=FactSource.AUDITOR,
            confidence=0.85,
            sequence=3,
            prior_value=False,
        )
        d = f.to_dict()
        f2 = Fact.from_dict(d)
        assert f2.name == f.name
        assert f2.value == f.value
        assert f2.source == f.source
        assert f2.confidence == f.confidence
        assert f2.sequence == f.sequence
        assert f2.prior_value == f.prior_value

    def test_from_dict_with_unknown_source_defaults_to_initial(self) -> None:
        f = Fact.from_dict({"name": "x", "value": 7, "source": "unknown_source"})
        assert f.source == FactSource.INITIAL


# ===========================================================================
# WorldState
# ===========================================================================


class TestWorldState:
    def test_empty(self) -> None:
        ws = WorldState()
        assert ws.facts == {}
        assert ws.resources == {}
        assert ws.temporal_state == {}
        assert ws.metadata == {}

    def test_get_fact_and_set_fact(self) -> None:
        ws = WorldState()
        assert ws.get_fact("x") is None
        f = Fact(name="x", value=True)
        ws.set_fact(f)
        assert ws.get_fact("x") is f
        assert ws.get_fact("x").value is True  # type: ignore[union-attr]

    def test_set_fact_preserves_prior_value(self) -> None:
        ws = WorldState()
        f1 = Fact(name="x", value=1, sequence=1)
        ws.set_fact(f1)
        f2 = Fact(name="x", value=2, sequence=2)
        ws.set_fact(f2)
        assert ws.get_fact("x").value == 2  # type: ignore[union-attr]
        assert ws.get_fact("x").prior_value == 1  # type: ignore[union-attr]

    def test_set_fact_value_convenience(self) -> None:
        ws = WorldState()
        f = ws.set_fact_value("ready", True, source=FactSource.STEP_STATE, sequence=1)
        assert f.name == "ready"
        assert f.value is True
        assert f.source == FactSource.STEP_STATE
        assert ws.get_fact("ready") is f

    def test_merge_overwrites_on_higher_sequence(self) -> None:
        ws1 = WorldState()
        ws1.set_fact(Fact(name="a", value=1, sequence=1))

        ws2 = WorldState()
        ws2.set_fact(Fact(name="a", value=2, sequence=2))
        ws2.set_fact(Fact(name="b", value=3, sequence=1))

        ws1.merge(ws2)
        assert ws1.get_fact("a").value == 2  # type: ignore[union-attr]  # overwritten (higher seq)
        assert ws1.get_fact("b").value == 3  # type: ignore[union-attr]  # new

    def test_merge_does_not_overwrite_on_lower_sequence(self) -> None:
        ws1 = WorldState()
        ws1.set_fact(Fact(name="a", value=2, sequence=2))

        ws2 = WorldState()
        ws2.set_fact(Fact(name="a", value=1, sequence=1))

        ws1.merge(ws2)
        assert ws1.get_fact("a").value == 2  # type: ignore[union-attr]  # not overwritten (lower seq)

    def test_merge_resources_and_temporal(self) -> None:
        ws1 = WorldState(resources={"budget": 10})
        ws2 = WorldState(resources={"budget": 5}, temporal_state={"step": 3})
        ws1.merge(ws2)
        assert ws1.resources["budget"] == 5
        assert ws1.temporal_state["step"] == 3

    def test_copy_is_independent(self) -> None:
        ws = WorldState()
        ws.set_fact(Fact(name="a", value=1))
        ws2 = ws.copy()
        ws2.set_fact(Fact(name="a", value=2))
        assert ws.get_fact("a").value == 1  # type: ignore[union-attr]  # original unchanged
        assert ws2.get_fact("a").value == 2  # type: ignore[union-attr]

    def test_roundtrip_dict(self) -> None:
        ws = WorldState(
            facts={"x": Fact(name="x", value=True, sequence=1)},
            resources={"budget": 100},
            temporal_state={"step_count": 5},
            metadata={"key": "val"},
        )
        d = ws.to_dict()
        ws2 = WorldState.from_dict(d)
        assert ws2.get_fact("x").value is True  # type: ignore[union-attr]
        assert ws2.resources["budget"] == 100
        assert ws2.temporal_state["step_count"] == 5
        assert ws2.metadata["key"] == "val"

    def test_from_dict_handles_raw_fact_objects(self) -> None:
        """WorldState.from_dict accepts either dicts or Fact objects."""
        ws = WorldState.from_dict({
            "facts": {"x": Fact(name="x", value=42)},
            "resources": {},
            "temporal_state": {},
            "metadata": {},
        })
        assert ws.get_fact("x").value == 42  # type: ignore[union-attr]

    def test_summary(self) -> None:
        ws = WorldState()
        ws.set_fact(Fact(name="a", value=True))
        ws.set_fact(Fact(name="b", value=False))
        ws.set_fact(Fact(name="c", value=None))
        ws.temporal_state["step_count"] = 3
        s = ws.summary()
        assert "SAT=['a']" in s
        assert "UNSAT=['b']" in s
        assert "UNKNOWN=['c']" in s
        assert "step=3" in s

    def test_to_quro_thinking_requires_quro_thinking(self) -> None:
        """If quro-thinking is not installed, to_quro_thinking raises
        ImportError.  If it is installed, it returns a quro-thinking
        WorldState."""
        ws = WorldState()
        ws.set_fact(Fact(name="x", value=True, sequence=1))
        try:
            qt_ws = ws.to_quro_thinking()
            # If we reach here, quro-thinking is installed.
            assert qt_ws.facts["x"].value is True
            assert qt_ws.facts["x"].sequence == 1
        except ImportError:
            # quro-thinking not installed — expected in CI without the dep.
            pass

    def test_from_quro_thinking_roundtrip(self) -> None:
        """If quro-thinking is available, roundtrip through from_quro_thinking."""
        try:
            from quro_thinking.domains.planning.state import (
                Fact as QTFact,
                WorldState as QTWorldState,
            )

            qt_ws = QTWorldState(
                facts={"x": QTFact(name="x", value=42, sequence=5)},
                resources={"r": 10},
            )
            ws = WorldState.from_quro_thinking(qt_ws)
            assert ws.get_fact("x").value == 42  # type: ignore[union-attr]
            assert ws.resources["r"] == 10
        except ImportError:
            pass


# ===========================================================================
# FactExtractor
# ===========================================================================


class TestFactExtractor:
    def test_extract_from_step_basic(self) -> None:
        ext = FactExtractor()
        sr = StepResult.success(
            step_id="step1",
            state={"tests_pass": True, "lint_ok": False},
            artifacts=[],
        )
        ws = ext.extract_from_step(sr)
        assert ws.get_fact("step1_ok").value is True  # type: ignore[union-attr]
        assert ws.get_fact("step1_artifact_count").value == 0  # type: ignore[union-attr]
        assert ws.get_fact("tests_pass").value is True  # type: ignore[union-attr]
        assert ws.get_fact("lint_ok").value is False  # type: ignore[union-attr]

    def test_extract_from_step_failure(self) -> None:
        ext = FactExtractor()
        sr = StepResult.failure(step_id="step1", error="boom", state={"retries": 3})
        ws = ext.extract_from_step(sr)
        assert ws.get_fact("step1_ok").value is False  # type: ignore[union-attr]
        assert ws.get_fact("retries").value == 3  # type: ignore[union-attr]

    def test_extract_with_prefix(self) -> None:
        ext = FactExtractor(fact_prefix="pipeline")
        sr = StepResult.success(step_id="step1", state={"done": True})
        ws = ext.extract_from_step(sr)
        assert ws.get_fact("pipeline.step1_ok").value is True  # type: ignore[union-attr]

    def test_extract_from_step_with_artifacts(self) -> None:
        ext = FactExtractor()
        sr = StepResult.success(
            step_id="step1",
            state={},
            artifacts=[
                {"status": "ok", "summary": "all good"},
                {"result": 42},
            ],
        )
        ws = ext.extract_from_step(sr)
        assert ws.get_fact("step1_artifact_count").value == 2  # type: ignore[union-attr]
        assert ws.get_fact("step1_artifact_0_status").value == "ok"  # type: ignore[union-attr]
        assert ws.get_fact("step1_artifact_1_result").value == 42  # type: ignore[union-attr]

    def test_extract_from_state_is_none(self) -> None:
        """state=None should not crash."""
        ext = FactExtractor()
        sr = StepResult(step_id="step1", ok=True, state=None)
        ws = ext.extract_from_step(sr)
        assert ws.get_fact("step1_ok").value is True  # type: ignore[union-attr]

    def test_extract_from_pipeline(self) -> None:
        ext = FactExtractor()
        sr1 = StepResult.success(step_id="s1", state={"a": True})
        sr2 = StepResult.success(step_id="s2", state={"b": False})
        pr = PipelineResult(
            ok=True,
            step_results={"s1": sr1, "s2": sr2},
            order=["s1", "s2"],
            partial=False,
        )
        ws = ext.extract_from_pipeline(pr)
        assert ws.get_fact("pipeline_ok").value is True  # type: ignore[union-attr]
        assert ws.get_fact("pipeline_partial").value is False  # type: ignore[union-attr]
        assert ws.get_fact("a").value is True  # type: ignore[union-attr]
        assert ws.get_fact("b").value is False  # type: ignore[union-attr]
        assert ws.temporal_state["step_count"] == 2

    def test_extract_from_pipeline_with_error(self) -> None:
        ext = FactExtractor()
        pr = PipelineResult(
            ok=False,
            step_results={},
            order=[],
            error="validation failed",
            replan_reason="step missing",
        )
        ws = ext.extract_from_pipeline(pr)
        assert ws.metadata["pipeline_error"] == "validation failed"
        assert ws.metadata["replan_reason"] == "step missing"


# ===========================================================================
# compute_goal_status
# ===========================================================================


class TestComputeGoalStatus:
    def test_all_sat(self) -> None:
        ws = WorldState()
        ws.set_fact(Fact(name="a", value=True))
        ws.set_fact(Fact(name="b", value=True))
        d = compute_goal_status(ws, ["a", "b"])
        assert d.is_sat
        assert d.sat_facts == ["a", "b"]
        assert d.unsat_facts == []

    def test_some_unsat(self) -> None:
        ws = WorldState()
        ws.set_fact(Fact(name="a", value=True))
        ws.set_fact(Fact(name="b", value=False))
        d = compute_goal_status(ws, ["a", "b", "c"])
        assert not d.is_sat
        assert d.sat_facts == ["a"]
        assert set(d.unsat_facts) == {"b", "c"}  # c is unknown → unsat

    def test_fact_missing_is_unsat(self) -> None:
        ws = WorldState()
        ws.set_fact(Fact(name="a", value=True))
        d = compute_goal_status(ws, ["a", "missing"])
        assert not d.is_sat
        assert "missing" in d.unsat_facts

    def test_empty_goals_is_sat(self) -> None:
        ws = WorldState()
        d = compute_goal_status(ws, [])
        assert d.is_sat
        assert d.unsat_facts == []


# ===========================================================================
# UNSATDiagnosis
# ===========================================================================


class TestUNSATDiagnosis:
    def test_sat_classmethod(self) -> None:
        d = UNSATDiagnosis.sat()
        assert d.is_sat
        assert d.unsat_facts == []

    def test_is_sat_true(self) -> None:
        d = UNSATDiagnosis(level=UNSATLevel.VERIFY_UNSAT, unsat_facts=[])
        assert d.is_sat

    def test_is_sat_false_for_run_unsat(self) -> None:
        d = UNSATDiagnosis(level=UNSATLevel.RUN_UNSAT, unsat_facts=[])
        assert not d.is_sat

    def test_is_terminal_unsat(self) -> None:
        d = UNSATDiagnosis(level=UNSATLevel.PROOF_UNSAT)
        assert d.is_terminal_unsat
        d2 = UNSATDiagnosis(level=UNSATLevel.VERIFY_UNSAT, unsat_facts=["x"])
        assert not d2.is_terminal_unsat

    def test_roundtrip_dict(self) -> None:
        d = UNSATDiagnosis(
            level=UNSATLevel.VERIFY_UNSAT,
            unsat_facts=["a"],
            sat_facts=["b"],
            cause="not done",
            step_id="step1",
            step_error="boom",
            proof_type="state_trap",
            exhausted_checkpoints=["cp1"],
            disposition=Disposition.ADD_STEPS,
            round_index=2,
            stall_count=1,
        )
        d2 = UNSATDiagnosis.from_dict(d.to_dict())
        assert d2.level == d.level
        assert d2.unsat_facts == d.unsat_facts
        assert d2.sat_facts == d.sat_facts
        assert d2.cause == d.cause
        assert d2.disposition == d.disposition
        assert d2.stall_count == d.stall_count


# ===========================================================================
# classify_unsat — the four levels
# ===========================================================================


class TestClassifyUnsat:
    def _base(self) -> UNSATDiagnosis:
        return UNSATDiagnosis(level=UNSATLevel.VERIFY_UNSAT, unsat_facts=["x"])

    # -- PROOF-UNSAT ---------------------------------------------------------

    def test_proof_unsat(self) -> None:
        d = classify_unsat(
            self._base(),
            proof_evidence={"type": "resource_exhaustion", "resource": "budget", "current_value": 0},
        )
        assert d.level == UNSATLevel.PROOF_UNSAT
        assert d.disposition == Disposition.ABORT

    # -- PLAN-UNSAT ----------------------------------------------------------

    def test_plan_unsat_first_round(self) -> None:
        d = classify_unsat(self._base(), pef_empty=True, round_index=0)
        assert d.level == UNSATLevel.PLAN_UNSAT
        assert d.disposition == Disposition.CONTINUE

    def test_plan_unsat_after_retry(self) -> None:
        d = classify_unsat(self._base(), pef_empty=True, round_index=2)
        assert d.level == UNSATLevel.PLAN_UNSAT
        assert d.disposition == Disposition.ESCALATE_L1

    # -- RUN-UNSAT -----------------------------------------------------------

    def test_run_unsat(self) -> None:
        d = classify_unsat(self._base(), step_error="crash", step_id="s1")
        assert d.level == UNSATLevel.RUN_UNSAT
        assert d.disposition == Disposition.RETRY_STEP

    # -- VERIFY-UNSAT with stall detection ----------------------------------

    def test_verify_unsat_no_stall(self) -> None:
        d = classify_unsat(
            self._base(),
            previous_unsat_facts=["y"],  # different from current ["x"]
            round_index=1,
        )
        assert d.level == UNSATLevel.VERIFY_UNSAT
        assert d.disposition == Disposition.ADD_STEPS
        assert d.stall_count == 0

    def test_verify_unsat_stalled(self) -> None:
        d = classify_unsat(
            self._base(),
            previous_unsat_facts=["x"],  # same as current
            round_index=2,
            max_stall_rounds=1,
        )
        assert d.level == UNSATLevel.VERIFY_UNSAT
        assert d.disposition == Disposition.ESCALATE_L1
        assert d.stall_count >= 1

    # -- SAT -----------------------------------------------------------------

    def test_sat(self) -> None:
        d = classify_unsat(UNSATDiagnosis.sat())
        assert d.level == UNSATLevel.VERIFY_UNSAT
        assert d.unsat_facts == []
        assert d.is_sat
        assert d.disposition == Disposition.CONTINUE

    # -- priority: PROOF > PLAN > RUN > VERIFY -------------------------------

    def test_proof_takes_priority_over_plan(self) -> None:
        d = classify_unsat(
            self._base(),
            pef_empty=True,
            proof_evidence={"type": "state_trap"},
        )
        assert d.level == UNSATLevel.PROOF_UNSAT


# ===========================================================================
# goal_status_from_results — evaluate stage entry point
# ===========================================================================


class TestGoalStatusFromResults:
    def test_sat_pipeline(self) -> None:
        sr = StepResult.success(step_id="s1", state={"a": True, "b": True})
        pr = PipelineResult(ok=True, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(pr, goal_facts=["a", "b"], round_index=0)
        assert d.is_sat
        assert d.sat_facts == ["a", "b"]

    def test_run_unsat_from_step_failure(self) -> None:
        sr = StepResult.failure(step_id="s1", error="timeout")
        pr = PipelineResult(ok=False, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(pr, goal_facts=["a"], round_index=0)
        assert d.level == UNSATLevel.RUN_UNSAT
        assert d.step_id == "s1"

    def test_verify_unsat(self) -> None:
        sr = StepResult.success(step_id="s1", state={"a": False})
        pr = PipelineResult(ok=True, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(pr, goal_facts=["a"], round_index=0)
        assert d.level == UNSATLevel.VERIFY_UNSAT
        assert d.unsat_facts == ["a"]
        assert d.disposition == Disposition.ADD_STEPS

    def test_pef_empty_plan_unsat(self) -> None:
        sr = StepResult.success(step_id="s1", state={})
        pr = PipelineResult(ok=True, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(
            pr, goal_facts=["a"], pef_empty=True, round_index=1,
        )
        assert d.level == UNSATLevel.PLAN_UNSAT
        assert d.disposition == Disposition.ESCALATE_L1

    def test_proof_unsat(self) -> None:
        sr = StepResult.success(step_id="s1", state={})
        pr = PipelineResult(ok=True, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(
            pr,
            goal_facts=["a"],
            proof_evidence={"type": "resource_exhaustion", "reason": "out of memory"},
            round_index=0,
        )
        assert d.level == UNSATLevel.PROOF_UNSAT
        assert d.disposition == Disposition.ABORT

    def test_with_custom_extractor(self) -> None:
        """A custom extractor can be injected."""
        class NullExtractor:
            def extract_from_step(self, result: StepResult) -> WorldState:
                return WorldState()
            def extract_from_pipeline(self, result: PipelineResult) -> WorldState:
                return WorldState()

        sr = StepResult.success(step_id="s1", state={"a": True})
        pr = PipelineResult(ok=True, step_results={"s1": sr}, order=["s1"])
        d = goal_status_from_results(
            pr, goal_facts=["a"], extractor=NullExtractor(), round_index=0,
        )
        # Null extractor produces empty WorldState → "a" is unknown → UNSAT
        assert not d.is_sat
        assert "a" in d.unsat_facts


# ===========================================================================
# AuditorFactExtractor — placeholder
# ===========================================================================


class TestAuditorFactExtractor:
    def test_placeholder_raises(self) -> None:
        ext = AuditorFactExtractor()
        with pytest.raises(NotImplementedError):
            ext.extract(["tests_pass"], ["artifact summary"])


# ===========================================================================
# ProblemSignature
# ===========================================================================


class TestProblemSignature:
    def test_deterministic(self) -> None:
        s1 = ProblemSignature.compute("Build a web app", ["tests_pass", "lint_ok"])
        s2 = ProblemSignature.compute("Build a web app", ["lint_ok", "tests_pass"])
        assert s1.sig == s2.sig  # order-independent

    def test_different_problems_have_different_sigs(self) -> None:
        s1 = ProblemSignature.compute("Build a web app", ["tests_pass"])
        s2 = ProblemSignature.compute("Build a mobile app", ["tests_pass"])
        assert s1.sig != s2.sig

    def test_prefix_truncation(self) -> None:
        """Long problems are truncated to prefix_chars."""
        long_problem = "A" * 500 + " different suffix"
        short_problem = "A" * 500 + " another suffix"
        # Both have same first 200 chars → same sig when goal_facts match.
        s1 = ProblemSignature.compute(long_problem, ["a"])
        s2 = ProblemSignature.compute(short_problem, ["a"])
        assert s1.sig == s2.sig  # prefix is only 200 chars

    def test_to_memory_id(self) -> None:
        s = ProblemSignature.compute("problem", ["a"])
        assert s.to_memory_id(3) == f"{s.sig}:r3"

    def test_to_memory_tags(self) -> None:
        s = ProblemSignature.compute("problem", ["a"])
        tags = s.to_memory_tags()
        assert f"problem:{s.sig}" in tags


# ===========================================================================
# RoundRecord
# ===========================================================================


class TestRoundRecord:
    def _signature(self) -> ProblemSignature:
        return ProblemSignature.compute("test problem", ["a", "b"])

    def test_basic(self) -> None:
        sig = self._signature()
        diag = UNSATDiagnosis(
            level=UNSATLevel.VERIFY_UNSAT,
            unsat_facts=["a"],
            sat_facts=["b"],
            cause="not done",
            disposition=Disposition.ADD_STEPS,
        )
        rr = RoundRecord(
            signature=sig,
            round_index=2,
            diagnosis=diag,
            world_state_summary="step=3 SAT=['b'] UNSAT=['a']",
            completed_step_ids=["s1", "s2"],
            pipeline_error=None,
        )
        assert rr.memory_id == sig.to_memory_id(2)

    def test_memory_body(self) -> None:
        sig = self._signature()
        diag = UNSATDiagnosis(
            level=UNSATLevel.VERIFY_UNSAT,
            unsat_facts=["a"],
            sat_facts=["b"],
            cause="tests fail",
            disposition=Disposition.ADD_STEPS,
        )
        rr = RoundRecord(signature=sig, round_index=1, diagnosis=diag)
        body = rr.to_memory_body()
        assert "Round 1:" in body
        assert "UNSAT_facts: ['a']" in body
        assert "Cause: tests fail" in body

    def test_memory_tags_includes_pattern(self) -> None:
        sig = self._signature()
        diag = UNSATDiagnosis(level=UNSATLevel.PROOF_UNSAT)
        rr = RoundRecord(signature=sig, round_index=0, diagnosis=diag)
        tags = rr.to_memory_tags()
        assert "pattern:proof_unsat" in tags

    def test_roundtrip_dict(self) -> None:
        sig = self._signature()
        diag = UNSATDiagnosis(
            level=UNSATLevel.VERIFY_UNSAT,
            unsat_facts=["a"],
            cause="x",
        )
        rr = RoundRecord(
            signature=sig,
            round_index=0,
            diagnosis=diag,
            world_state_summary="ws",
            completed_step_ids=["s1"],
        )
        d = rr.to_dict()
        rr2 = RoundRecord.from_dict(d)
        assert rr2.round_index == rr.round_index
        assert rr2.diagnosis.level == rr.diagnosis.level
        assert rr2.world_state_summary == "ws"
