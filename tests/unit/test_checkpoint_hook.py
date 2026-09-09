"""Tests for the phase-2 checkpointing (stage 4)."""

from __future__ import annotations

from quro.core.domain.step_type import StepType, StepTypeCatalog
from quro.core.features import Feature
from quro.core.resources import (
    FileResourceStore,
    JobKey,
    ResourceRef,
    RuntimeSessionLedger,
    goal_uuid_from_facts,
)
from quro.steps.checkpoint_hook import (
    CheckpointHook,
    RoundCheckpoint,
    accumulate_state,
    restore_domain_state,
)
from quro.steps.core import StepResult, StepSpec
from quro.steps.hooks import HookContext


def _step(id: str, objective: str = "") -> StepSpec:
    return StepSpec(id=id, objective=objective or f"objective {id}")


def _store(tmp_path) -> FileResourceStore:
    return FileResourceStore(tmp_path / "session")


# -- accumulate_state -------------------------------------------------------


def test_accumulate_state_merges_steps_and_artifacts():
    results = {
        "s1": StepResult.success(
            "s1",
            state={"steps": [{"step_id": "s1", "status": "completed"}],
                   "artifacts": [{"artifact_id": "art_a", "step_id": "s1"}],
                   "plan": "P1"},
            artifacts=[{"artifact_id": "art_a", "step_id": "s1"}],
        ),
        "s2": StepResult.success(
            "s2",
            state={"steps": [{"step_id": "s2", "status": "completed"}],
                   "artifacts": [{"artifact_id": "art_b", "step_id": "s2"}],
                   "plan": "P2"},
            artifacts=[{"artifact_id": "art_b", "step_id": "s2"}],
        ),
    }
    merged = accumulate_state(results)
    assert [s["step_id"] for s in merged["steps"]] == ["s1", "s2"]
    assert [a["artifact_id"] for a in merged["artifacts"]] == ["art_a", "art_b"]
    assert merged["plan"] == "P2"  # last non-empty wins


# -- CheckpointHook ---------------------------------------------------------


def test_pre_step_snapshot_exists_before_execution(tmp_path):
    store = _store(tmp_path)
    hook = CheckpointHook(store, session_id="s1", round_idx=0)
    ctx = HookContext(problem="p", step_results={})

    # The snapshot must exist before the step executes (here: no execution).
    hook.on_pre_step(_step("s2"), ctx)
    assert store.load_snapshot(ResourceRef("step", "s2")) is not None


def test_pre_step_snapshot_contains_step_config(tmp_path):
    """Phase 4: the snapshot carries the config half alongside domain_state."""
    store = _store(tmp_path)
    catalog = StepTypeCatalog([
        StepType(
            "survey_module",
            skill_pool=["codegraph"],
            identity_block="identity survey",
            executor_hints={"phase": "survey"},
            features=(Feature.RECOVERY,),
        ),
    ])
    hook = CheckpointHook(store, session_id="s1", round_idx=0, step_types=catalog)
    ctx = HookContext(problem="p", step_results={})

    hook.on_pre_step(StepSpec(id="s2", step_type="survey_module", objective="map"), ctx)

    snapshot = store.load_snapshot(ResourceRef("step", "s2"))
    cfg = snapshot["step_config"]
    assert cfg["id"] == "s2"
    assert cfg["step_type"] == "survey_module"
    assert cfg["identity_block"] == "identity survey"
    assert cfg["executor_hints"]["phase"] == "survey"
    assert cfg["features"] == ["recovery"]


def test_step_config_without_catalog_stays_empty(tmp_path):
    """Without a StepType catalog, inlined fields are empty (no crash)."""
    store = _store(tmp_path)
    hook = CheckpointHook(store, session_id="s1", round_idx=0)
    ctx = HookContext(problem="p", step_results={})

    hook.on_pre_step(StepSpec(id="s3", step_type="survey_module"), ctx)

    cfg = store.load_snapshot(ResourceRef("step", "s3"))["step_config"]
    assert cfg["id"] == "s3"
    assert cfg["identity_block"] == ""
    assert cfg["features"] == []


def test_resume_restores_domain_state_after_interrupt(tmp_path):
    store = _store(tmp_path)
    hook = CheckpointHook(store, session_id="s1", round_idx=0)

    # s1 completed earlier and produced an artifact.
    prior = {
        "s1": StepResult.success(
            "s1",
            state={
                "steps": [{"step_id": "s1", "status": "completed"}],
                "artifacts": [{"artifact_id": "art_a", "step_id": "s1"}],
            },
            artifacts=[{"artifact_id": "art_a", "step_id": "s1"}],
        ),
    }
    ctx = HookContext(problem="p", step_results=prior)
    hook.on_pre_step(_step("s2"), ctx)

    # Simulated interrupt mid-s2: resume from the step checkpoint.
    restored = restore_domain_state(store, "s2")
    assert restored is not None

    # [c7] earlier steps' artifacts survive (hack-step without breaking-round).
    artifact_ids = [a["artifact_id"] for a in restored["artifacts"]]
    assert "art_a" in artifact_ids
    assert "s1" in [s["step_id"] for s in restored["steps"]]


def test_checkpoint_records_reasoning_reference(tmp_path):
    store = _store(tmp_path)
    hook = CheckpointHook(store, session_id="s1", round_idx=1)
    ctx = HookContext(problem="p", step_results={})

    hook.on_pre_step(_step("deep_dive"), ctx)

    snapshot = store.load_snapshot(ResourceRef("step", "deep_dive"))
    assert snapshot["reasoning_session_id"] == "s1-deep_dive"
    assert snapshot["round_idx"] == 1
    # The executor receives the stable reference + storage dir via metadata.
    assert ctx.metadata["reasoning_session_id"] == "s1-deep_dive"
    assert ctx.metadata["reasoning_storage_dir"] == str(tmp_path / "session" / "reasoning")


def test_restore_unknown_step_returns_none(tmp_path):
    store = _store(tmp_path)
    assert restore_domain_state(store, "never_ran") is None


def test_on_pre_step_writes_step_granularity_resume_point(tmp_path):
    """R0a-2 / G3: on_pre_step refines ledger current_round + current_step."""
    ledger = RuntimeSessionLedger(tmp_path)
    job = JobKey("mock", "resume-demo", goal_uuid_from_facts(["done"]))
    session_id, _ = ledger.create_session(job)
    store = FileResourceStore(ledger.session_dir(job, session_id))

    hook = CheckpointHook(
        store, session_id=session_id, round_idx=2, ledger=ledger, job=job,
    )
    ctx = HookContext(problem="p", step_results={})
    hook.on_pre_step(_step("s2"), ctx)

    data = ledger.load(job, session_id)
    assert data["current_round"] == 2
    assert data["current_step"] == "s2"


def test_on_pre_step_without_ledger_is_backward_compatible(tmp_path):
    """No ledger/job → on_pre_step writes the checkpoint but not the ledger."""
    store = _store(tmp_path)
    hook = CheckpointHook(store, session_id="s1", round_idx=0)
    ctx = HookContext(problem="p", step_results={})
    hook.on_pre_step(_step("s2"), ctx)
    assert store.load_snapshot(ResourceRef("step", "s2")) is not None


# -- RoundCheckpoint --------------------------------------------------------


def test_round_boundary_produces_one_start_and_one_end(tmp_path):
    store = _store(tmp_path)
    cp = RoundCheckpoint(store, session_id="s1")

    cp.round_start(0, "goal 0", {"steps": []})
    cp.round_end(0, {"sat_facts": ["goal_0"], "unsat_facts": []},
                 {"steps": [{"step_id": "s1"}]})

    start = store.load_snapshot(ResourceRef("round", "0", ("start",)))
    end = store.load_snapshot(ResourceRef("round", "0", ("end",)))
    assert start is not None
    assert end is not None
    assert start["goal"] == "goal 0"
    assert end["goal_status"] == {"sat_facts": ["goal_0"], "unsat_facts": []}

    # [c8] exactly one start and one end snapshot per round.
    assert cp.load_start(0) == start
    assert cp.load_end(0) == end
    # A different round has no snapshot yet.
    assert store.load_snapshot(ResourceRef("round", "1", ("start",))) is None


def test_round_checkpoints_are_independent_per_round(tmp_path):
    store = _store(tmp_path)
    cp = RoundCheckpoint(store, session_id="s1")

    cp.round_start(0, "g0", None)
    cp.round_end(0, {}, None)
    cp.round_start(1, "g1", {"steps": [{"step_id": "s0"}]})
    cp.round_end(1, {"sat_facts": ["g1"], "unsat_facts": []}, None)

    assert cp.load_start(0)["goal"] == "g0"
    assert cp.load_start(1)["goal"] == "g1"
    assert cp.load_start(1)["domain_state"]["steps"][0]["step_id"] == "s0"
