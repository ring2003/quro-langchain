"""Tests for the phase-2 runtime-session ledger (stage 3)."""

from __future__ import annotations

import json

from quro.core.resources import (
    JobKey,
    RuntimeSessionLedger,
    goal_uuid_from_facts,
)


def _job() -> JobKey:
    return JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["tests_pass"]))


# -- schema / round-trip ----------------------------------------------------


def test_new_ledger_has_core_fields():
    job = _job()
    ledger = RuntimeSessionLedger.new_ledger(job, "s1")
    assert ledger["version"] == 1
    assert ledger["session_id"] == "s1"
    assert ledger["job"] == {
        "domain": "codebase_research",
        "problem_name": "parse-config",
        "goal_uuid": job.goal_uuid,
    }
    assert ledger["current_round"] == 0
    assert ledger["current_step"] == ""
    assert ledger["rounds"] == []
    assert ledger["extensions"] == {}


def test_ledger_round_trips(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, ledger = store.create_session(job)

    ledger["current_round"] = 3
    ledger["current_step"] = "step_ab12"
    store.save(job, ledger)

    reloaded = store.load(job, session_id)
    assert reloaded == ledger


def test_unknown_field_preserved_after_write(tmp_path):
    # [c5]: a ledger with an unknown "custom_metric" field reads back intact.
    store = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, ledger = store.create_session(job)

    ledger["custom_metric"] = {"importance": 0.9, "tags": ["a", "b"]}
    store.save(job, ledger)

    reloaded = store.load(job, session_id)
    assert reloaded["custom_metric"] == {"importance": 0.9, "tags": ["a", "b"]}

    # An update to a known field must not drop the unknown one.
    store.update(job, session_id, current_round=2)
    assert store.load(job, session_id)["custom_metric"] == {"importance": 0.9, "tags": ["a", "b"]}


def test_dynamic_member_with_extra_fields_round_trips(tmp_path):
    # Tier 3: rounds[]/members[] entries may carry arbitrary extra fields.
    store = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, ledger = store.create_session(job)

    ledger["rounds"] = [
        {
            "round_idx": 0,
            "status": "completed",
            "verdict": "sat",
            "goal": "g",
            "members": [
                {"type": "artifact", "ref": "artifact://s1/art_ab12", "custom_metric": {"x": 1}},
            ],
        }
    ]
    store.save(job, ledger)

    reloaded = store.load(job, session_id)
    member = reloaded["rounds"][0]["members"][0]
    assert member["type"] == "artifact"
    assert member["custom_metric"] == {"x": 1}


def test_version_field_preserved(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, ledger = store.create_session(job)

    assert ledger["version"] == 1
    ledger["version"] = 2  # simulate a bumped schema version
    store.save(job, ledger)

    assert store.load(job, session_id)["version"] == 2


# -- session id allocation (c6) --------------------------------------------


def test_fresh_run_allocates_incrementing_session_ids(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job = _job()

    s1, _ = store.create_session(job)
    s2, _ = store.create_session(job)
    s3, _ = store.create_session(job)

    assert (s1, s2, s3) == ("s1", "s2", "s3")


def test_resume_reuses_same_session_id(tmp_path):
    # [c6]: resume with the same job returns the same session_id.
    store = RuntimeSessionLedger(tmp_path)
    job = _job()

    s1, _ = store.create_session(job)
    s2, _ = store.create_session(job)

    resumed, _ = store.create_session(job, resume=True)
    assert resumed == s2  # latest, not a fresh id

    # A subsequent fresh run still gets the next incrementing id.
    s3, _ = store.create_session(job)
    assert s3 == "s3"


def test_resume_with_no_sessions_allocates_new(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job = _job()

    sid, _ = store.create_session(job, resume=True)
    assert sid == "s1"


def test_different_jobs_have_independent_session_ids(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job_a = _job()
    job_b = JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["other"]))

    sa, _ = store.create_session(job_a)
    sb, _ = store.create_session(job_b)

    assert sa == "s1"
    assert sb == "s1"  # independent incrementing under a different goal_uuid


def test_ledger_written_to_session_index_path(tmp_path):
    store = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, _ = store.create_session(job)

    expected = store.session_dir(job, session_id) / "index.json"
    assert expected.exists()
    assert json.loads(expected.read_text(encoding="utf-8"))["session_id"] == session_id
