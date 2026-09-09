"""Tests for the phase-3 ``CLIView`` (read-only projection).

Covers the view-layer acceptance criteria from
``unified-resource-layer-phase3-implementation-plan.md`` §6: job/session/round
rendering and dangling-snapshot placeholders (c1-c4), plus the unknown-job /
unknown-session error contract.
"""

from __future__ import annotations

import json

import pytest

from quro.cli.view import CLIView, JobNotFoundError, SessionNotFoundError
from quro.core.resources import (
    FileResourceStore,
    JobIndex,
    JobKey,
    RuntimeSessionLedger,
    goal_uuid_from_facts,
)
from quro.steps.checkpoint_hook import RoundCheckpoint


def _job() -> JobKey:
    return JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["a"]))


def _session(tmp_path, *, summary: str = "ran to round 1"):
    ledger = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, _ = ledger.create_session(job, summary=summary)
    store = FileResourceStore(ledger.session_dir(job, session_id))
    return job, session_id, ledger, store


# -- job level (c1) --------------------------------------------------------


def test_render_job_lists_sessions_with_id_created_summary(tmp_path):
    job, session_id, ledger, _ = _session(tmp_path)
    text = CLIView().render_job(job, JobIndex(tmp_path))
    assert f"Job: {job.descriptor()}" in text
    assert session_id in text
    assert "ran to round 1" in text


def test_render_job_unknown_job_raises(tmp_path):
    with pytest.raises(JobNotFoundError):
        CLIView().render_job(_job(), JobIndex(tmp_path))


def test_render_job_no_sessions_renders_none(tmp_path):
    index = JobIndex(tmp_path)
    job = _job()
    index.upsert(job, "s1", summary="")
    # Rewrite the index with an empty sessions list (job known, no runs).
    data = index.read(job)
    data["sessions"] = []
    (index.job_dir(job) / "index.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    assert "Sessions: (none)" in CLIView().render_job(job, index)


# -- session level (c2) ----------------------------------------------------


def test_render_session_shows_resume_point_and_round_goal_status(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    cp = RoundCheckpoint(store, session_id)
    cp.round_start(0, "parse the config", {})
    cp.round_end(0, {"sat_facts": ["parse_config_done"], "unsat_facts": []},
                 {"phase": "step_execute"})
    cp.round_start(1, "summarize the config", {})
    ledger.update(job, session_id, current_round=1, current_step="s2")

    text = CLIView().render_session(job, session_id, ledger, store)
    assert "current_round: 1" in text
    assert "current_step: s2" in text
    assert "round 0" in text and "goal_status=" in text
    assert "goal=parse the config" in text
    # round 1 has only a start snapshot → unknown goal_status, start goal shown.
    assert "round 1" in text
    assert "goal_status=(unknown)" in text
    assert "goal=summarize the config" in text


def test_render_session_missing_snapshot_renders_placeholder(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    ledger.update(job, session_id, current_round=1)
    text = CLIView().render_session(job, session_id, ledger, store)
    assert "(missing snapshot)" in text


def test_render_session_unknown_session_raises(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    with pytest.raises(SessionNotFoundError):
        CLIView().render_session(job, "s99", ledger, store)


# -- round level (c3) ------------------------------------------------------


def test_render_round_shows_steps_artifacts_phase(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    cp = RoundCheckpoint(store, session_id)
    state = {
        "steps": [
            {"step_id": "s1", "status": "completed"},
            {"step_id": "s2", "status": "in_progress"},
        ],
        "artifacts": [
            {"artifact_id": "art_s1", "summary": "parsed config", "kind": "analysis"},
        ],
        "phase": "step_execute",
        "executing_step_id": "s2",
    }
    cp.round_start(0, "parse the config", {})
    cp.round_end(0, {"sat_facts": ["parse_config_done"], "unsat_facts": []}, state)

    text = CLIView().render_round(store, 0)
    assert "Round: 0" in text
    assert "goal_status:" in text
    assert "phase: step_execute" in text
    assert "executing_step_id: s2" in text
    assert "s1  completed" in text
    assert "s2  in_progress" in text
    assert "art_s1" in text and "kind=analysis" in text


def test_render_round_offloaded_artifact_renders_without_crash(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    cp = RoundCheckpoint(store, session_id)
    # An offloaded artifact: DomainState holds descriptor + ref only (no kind).
    state = {
        "steps": [{"step_id": "s1", "status": "completed"}],
        "artifacts": [
            {"artifact_id": "art_s1", "summary": "parsed config",
             "step_id": "s1", "ref": "artifact://art_s1"},
        ],
        "phase": "step_execute",
        "executing_step_id": "",
    }
    cp.round_end(0, {}, state)
    text = CLIView().render_round(store, 0)
    assert "art_s1" in text
    assert "kind=" not in text  # payload not materialised; kind stays absent


def test_render_round_missing_snapshot_renders_placeholder(tmp_path):
    job, session_id, ledger, store = _session(tmp_path)
    text = CLIView().render_round(store, 0)
    assert "Round: 0" in text
    assert "(missing snapshot)" in text
