"""Tests for the phase-3 ``quro`` CLI entry point (``main``).

Covers the CLI-level acceptance criteria from
``unified-resource-layer-phase3-implementation-plan.md`` §6: unknown-job →
clean message + non-zero exit (c1), session/round descent (c2/c3), the
``--round`` requires ``--session`` guard, and the no-kernel-import rule (c5).
"""

from __future__ import annotations

from pathlib import Path

from quro.cli import main
from quro.core.resources import (
    FileResourceStore,
    JobKey,
    RuntimeSessionLedger,
    goal_uuid_from_facts,
)
from quro.steps.checkpoint_hook import RoundCheckpoint


def _job() -> JobKey:
    return JobKey("codebase_research", "parse-config", goal_uuid_from_facts(["a"]))


def _setup(tmp_path, *, summary: str = "ran to round 2"):
    ledger = RuntimeSessionLedger(tmp_path)
    job = _job()
    session_id, _ = ledger.create_session(job, summary=summary)
    store = FileResourceStore(ledger.session_dir(job, session_id))
    return job, session_id, ledger, store


# -- error contract --------------------------------------------------------


def test_show_unknown_job_prints_message_and_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    code = main(["show", _job().descriptor()])
    captured = capsys.readouterr()
    assert code == 1
    assert "no such job" in captured.err


def test_show_invalid_descriptor_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    code = main(["show", "job://bad"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error" in captured.err


def test_show_round_requires_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    code = main(["show", _job().descriptor(), "--round", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--round requires --session" in captured.err


def test_show_unknown_session_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    job, _, _, _ = _setup(tmp_path)
    code = main(["show", job.descriptor(), "--session", "s99"])
    captured = capsys.readouterr()
    assert code == 1
    assert "no such session" in captured.err


# -- happy path ------------------------------------------------------------


def test_show_job_lists_sessions(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    job, session_id, _, _ = _setup(tmp_path)

    code = main(["show", job.descriptor()])
    captured = capsys.readouterr()
    assert code == 0
    assert f"Job: {job.descriptor()}" in captured.out
    assert session_id in captured.out
    assert "ran to round 2" in captured.out


def test_show_session_and_round_descend(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QURO_STATE_DIR", str(tmp_path))
    job, session_id, ledger, store = _setup(tmp_path)
    cp = RoundCheckpoint(store, session_id)
    cp.round_start(0, "parse the config", {})
    cp.round_end(0, {"sat_facts": ["parse_config_done"], "unsat_facts": []}, {
        "steps": [{"step_id": "s1", "status": "completed"}],
        "artifacts": [{"artifact_id": "art_s1", "summary": "x", "kind": "analysis"}],
        "phase": "step_execute",
        "executing_step_id": "",
    })
    ledger.update(job, session_id, current_round=0, current_step="s1")

    code = main(["show", job.descriptor(), "--session", session_id])
    captured = capsys.readouterr()
    assert code == 0
    assert "current_round: 0" in captured.out
    assert "current_step: s1" in captured.out
    assert "goal_status=" in captured.out

    code = main(["show", job.descriptor(), "--session", session_id, "--round", "0"])
    captured = capsys.readouterr()
    assert code == 0
    assert "s1  completed" in captured.out
    assert "art_s1" in captured.out


# -- c5: no kernel import --------------------------------------------------


def test_cli_and_view_do_not_import_kernel():
    src = Path(__file__).resolve().parents[2] / "src" / "quro"
    for rel in ("cli/__init__.py", "cli/view.py"):
        text = (src / rel).read_text(encoding="utf-8")
        assert "quro_thinking" not in text, f"{rel} imports quro_thinking"
