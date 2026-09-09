"""Tests for the phase-2 job key + job index (stage 2)."""

from __future__ import annotations

import pytest

from quro.core.resources import (
    InvalidJobDescriptor,
    JobIndex,
    JobKey,
    goal_uuid_from_facts,
    parse_job_descriptor,
    validate_problem_name,
)


# -- problem_name slug ------------------------------------------------------


@pytest.mark.parametrize("name", [
    "parse-config",
    "a",
    "codebase-research-2026",
    "x" * 63,
    "abc123",
])
def test_problem_name_accepts_valid_slugs(name):
    assert validate_problem_name(name) == name


@pytest.mark.parametrize("name", [
    "",
    "ParseConfig",       # uppercase
    "-leading",          # leading hyphen
    "has space",
    "under_score",
    "x" * 64,            # too long (max 63)
    "a/b",               # path separator
    None,
])
def test_problem_name_rejects_invalid_slugs(name):
    with pytest.raises(InvalidJobDescriptor):
        validate_problem_name(name)


def test_job_key_fails_fast_on_bad_slug():
    with pytest.raises(InvalidJobDescriptor):
        JobKey("codebase_research", "ParseConfig", "a1b2c3d4")


# -- goal_uuid --------------------------------------------------------------


def test_goal_uuid_is_deterministic_and_order_insensitive():
    assert goal_uuid_from_facts(["tests_pass", "code_compiles"]) == \
        goal_uuid_from_facts(["code_compiles", "tests_pass"])


def test_goal_uuid_is_16_hex_chars():
    uid = goal_uuid_from_facts(["a", "b"])
    assert len(uid) == 16
    assert all(c in "0123456789abcdef" for c in uid)


def test_changed_goal_facts_produce_new_uuid():
    uid1 = goal_uuid_from_facts(["tests_pass"])
    uid2 = goal_uuid_from_facts(["tests_pass", "code_compiles"])
    uid3 = goal_uuid_from_facts(["code_compiles"])
    assert uid1 != uid2
    assert uid2 != uid3


def test_goal_uuid_ignores_problem_text():
    # [c3]: a changed problem text with the same problem_name does not change
    # the goal_uuid — the uuid derives only from goal_facts.
    facts = ["tests_pass"]
    assert goal_uuid_from_facts(facts) == goal_uuid_from_facts(facts)
    # problem text is simply not an input to goal_uuid_from_facts.
    JobKey("codebase_research", "parse-config", goal_uuid_from_facts(facts))


# -- parse_job_descriptor ---------------------------------------------------


def test_parse_job_descriptor_round_trips():
    key = parse_job_descriptor("job://codebase_research/parse-config/a1b2c3d4")
    assert key == JobKey("codebase_research", "parse-config", "a1b2c3d4")
    assert key.descriptor() == "job://codebase_research/parse-config/a1b2c3d4"


@pytest.mark.parametrize("bad", [
    "job://only-two-parts",
    "job://a/b/c/d",
    "job://a//c",
    "http://a/b/c",
    "job://a/Bad/c",
])
def test_parse_job_descriptor_rejects_bad_syntax(bad):
    with pytest.raises(InvalidJobDescriptor):
        parse_job_descriptor(bad)


# -- JobIndex ---------------------------------------------------------------


def test_job_index_lists_sessions_and_distinguishes_jobs(tmp_path):
    index = JobIndex(tmp_path)
    job_a = JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["a"]))
    job_b = JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["b"]))

    index.upsert(job_a, "s1", summary="ran to round 2")
    index.upsert(job_a, "s2", summary="fresh run")
    index.upsert(job_b, "s1", summary="different goal")

    # Same job, different sessions → both listed under job_a.
    sessions_a = index.list_sessions(job_a)
    assert [s["session_id"] for s in sessions_a] == ["s1", "s2"]

    # Different job → different directory/index.
    sessions_b = index.list_sessions(job_b)
    assert [s["session_id"] for s in sessions_b] == ["s1"]

    # Physically separate job directories.
    assert index.job_dir(job_a) != index.job_dir(job_b)


def test_job_index_upsert_updates_summary_keeps_created_at(tmp_path):
    index = JobIndex(tmp_path)
    job = JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["a"]))

    index.upsert(job, "s1", summary="first", created_at=100.0)
    index.upsert(job, "s1", summary="updated")

    sessions = index.list_sessions(job)
    assert len(sessions) == 1
    assert sessions[0]["summary"] == "updated"
    assert sessions[0]["created_at"] == 100.0


def test_job_index_unknown_job_returns_empty(tmp_path):
    index = JobIndex(tmp_path)
    job = JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["a"]))
    assert index.list_sessions(job) == []
    assert index.read(job) is None
