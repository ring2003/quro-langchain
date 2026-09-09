"""Tests for Phase D — the codebase research domain (content + recovery ops)."""

from __future__ import annotations

from quro.core.tools.capability import Readonly, Shell, Write
from quro.domains.codebase_research import (
    RESEARCH_PHASES,
    CodebaseResearchDomain,
    default_skills,
    default_step_types,
)
from quro.runtime.backtracker import Backtracker, RoundObject


# ---------------------------------------------------------------------------
# Capability catalog
# ---------------------------------------------------------------------------


def test_domain_catalog():
    d = CodebaseResearchDomain()
    assert d.name == "codebase_research"
    assert d.phases() == ["survey", "narrow", "deep_dive", "synthesize", "assemble_report"]
    assert d.step_types.names() == [
        "survey_module", "deep_dive_symbol", "assemble_report",
    ]
    assert set(d.skills.names()) == {
        "codegraph", "read-strategy", "evidence-format", "report-template",
    }
    # Phase 4: recovery enablement is declared via StepType.features.
    assert d.step_types.features_for("survey_module") == ("recovery",)
    assert d.step_types.features_for("deep_dive_symbol") == ("recovery",)
    assert d.step_types.features_for("assemble_report") == ()
    # Tool grants are declared via StepType.tools as capability markers
    # (step-anchored tool governance — external grants only; internal semantic
    # ops are the domain ``apply`` op table's concern, not ``tools``).
    assert d.step_types.tools_for("survey_module") == (Readonly, Shell)
    assert d.step_types.tools_for("deep_dive_symbol") == (Readonly, Shell)
    assert d.step_types.tools_for("assemble_report") == (Readonly, Write)


def test_default_recovery_declaration_targets_explorer_steps():
    """InlineStep compaction recovery (§6) is declared only for the explorers."""
    d = CodebaseResearchDomain()
    assert d.recovery_mode == "compaction"
    assert d.recovery_compaction is None  # no domain-wide fallback
    assert d.recovery_compaction_for_step_type == {
        "survey_module": "backtrack_plan",
        "deep_dive_symbol": "backtrack_plan",
    }
    # The reporter step (assemble_report) is absent → plain structural resume.
    assert "assemble_report" not in d.recovery_compaction_for_step_type


def test_recovery_declaration_overridable():
    """The launcher may override the recovery declaration at construction."""
    d = CodebaseResearchDomain(
        recovery_mode="none",
        recovery_compaction="next_step",
        recovery_compaction_for_step_type={"survey_module": "next_step"},
    )
    assert d.recovery_mode == "none"
    assert d.recovery_compaction == "next_step"
    assert d.recovery_compaction_for_step_type == {"survey_module": "next_step"}


def test_phase_for_step_type():
    d = CodebaseResearchDomain()
    assert d.phase_for_step_type("survey_module") == "survey"
    assert d.phase_for_step_type("deep_dive_symbol") == "deep_dive"
    assert d.phase_for_step_type("assemble_report") == "assemble_report"
    assert d.phase_for_step_type("unknown") == "survey"


def test_step_type_skill_pools():
    catalog = default_step_types()
    assert catalog.skill_pool_for("survey_module") == ["codegraph", "read-strategy"]
    assert catalog.skill_pool_for("assemble_report") == ["report-template"]


def test_skills_load():
    skills = default_skills()
    assert skills.has("codegraph")
    body = skills.body_of("codegraph")
    assert body is not None and "codegraph query" in body


# ---------------------------------------------------------------------------
# State shape + evidence ops
# ---------------------------------------------------------------------------


def test_initial_state_shape():
    d = CodebaseResearchDomain()
    st = d.initial_state({"problem": "p", "phase": "deep_dive"})
    assert st["phase"] == "deep_dive"
    assert st["problem"] == "p"
    assert st["module_inventory"] == []
    assert st["report_chunks"] == []
    assert st["recovery"] == {}


def test_initial_state_bad_phase_falls_back_to_survey():
    d = CodebaseResearchDomain()
    assert d.initial_state({"phase": "bogus"})["phase"] == "survey"


def test_assemble_report_records_path_and_chunks():
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "assemble_report"})
    st, resp = d.apply(st, "assemble_report", {"path": "/tmp/r.md", "title": "t"})
    assert resp.ok
    assert st["report_path"] == "/tmp/r.md"
    assert st["report_chunks"] == ["/tmp/r.md"]
    assert d.is_complete(st) is False  # no artifacts yet


# ---------------------------------------------------------------------------
# Recovery journal ops
# ---------------------------------------------------------------------------


def _state_with_step(phase="deep_dive", step_id="s1"):
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": phase})
    st["executing_step_id"] = step_id
    return d, st


def test_commit_clue_appends_to_journal():
    d, st = _state_with_step()
    st, resp = d.apply(st, "commit_clue", {"clue": "found X"})
    assert resp.ok
    journal = st["recovery"]["s1"]
    assert journal["step_round"] == 0
    assert [c["text"] for c in journal["clues"]] == ["found X"]

    st, resp = d.apply(st, "commit_clue", {"clue": "and Y"})
    assert resp.ok
    assert [c["text"] for c in st["recovery"]["s1"]["clues"]] == ["found X", "and Y"]


def test_commit_clue_requires_clue():
    d, st = _state_with_step()
    _, resp = d.apply(st, "commit_clue", {"clue": "  "})
    assert not resp.ok


def test_continue_step_bumps_round_and_records_reason():
    d, st = _state_with_step()
    st, resp = d.apply(st, "continue_step", {"reason": "need callers"})
    assert resp.ok
    journal = st["recovery"]["s1"]
    assert journal["step_round"] == 1
    assert journal["continue_reasons"] == ["need callers"]


def test_commit_clue_defaults_to_executing_step():
    d, st = _state_with_step()
    st, resp = d.apply(st, "commit_clue", {"clue": "c", "step_id": ""})
    assert resp.ok
    assert "s1" in st["recovery"]


# ---------------------------------------------------------------------------
# Phase gating
# ---------------------------------------------------------------------------


def test_phase_gating_blocks_wrong_phase_op():
    d, st = _state_with_step(phase="deep_dive")
    _, resp = d.apply(st, "assemble_report", {"path": "/tmp/r.md"})
    assert not resp.ok
    assert "PHASE_MISMATCH" in resp.error


def test_explore_ops_allowed_in_all_explore_phases():
    d = CodebaseResearchDomain()
    for phase in RESEARCH_PHASES[:4]:  # survey/narrow/deep_dive/synthesize
        allowed = d.allowed_ops_for_phase(phase)
        assert "commit_clue" in allowed
        assert "add_artifact" in allowed


# ---------------------------------------------------------------------------
# complete_step + completion
# ---------------------------------------------------------------------------


def test_complete_step_requires_artifact():
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "deep_dive"})
    st["executing_step_id"] = "s1"
    st["steps"] = [{"step_id": "s1", "status": "in_progress", "depends_on": []}]
    _, resp = d.apply(st, "complete_step", {"step_id": "s1"})
    assert not resp.ok
    assert "no artifacts" in resp.error


def test_complete_step_steered_keeps_executing_step_id():
    """Steered steps: complete_step writes round_status only; step_status stays
    L1-owned (minted by the fold after gate.verify_completion).  Pipeline
    advancement keyed on step_status therefore keeps executing_step_id set
    across every L0 round — the single-step steering case that previously
    cleared it and denied the next round's add_artifact / complete_step with
    'no executing step — cannot resolve a principal'."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "survey"})
    st["executing_step_id"] = "s1"
    st["steps"] = [{
        "step_id": "s1",
        "step_type": "survey_module",
        "status": "in_progress",
        "depends_on": [],
    }]
    st["artifacts"] = [{"artifact_id": "a1", "step_id": "s1", "summary": "evidence"}]
    new_state, resp = d.apply(st, "complete_step", {"step_id": "s1"})
    assert resp.ok
    s1 = new_state["steps"][0]
    assert s1["round_status"] == "completed"
    assert s1.get("step_status") != "completed"
    assert new_state["executing_step_id"] == "s1"


def test_complete_step_ordinary_mints_both_tokens_and_advances():
    """Ordinary (non-steered) steps: L0 == L1, so complete_step mints both
    round_status and step_status; the last completed step clears
    executing_step_id so the pipeline can advance."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "assemble_report"})
    st["executing_step_id"] = "s1"
    st["steps"] = [{
        "step_id": "s1",
        "step_type": "assemble_report",
        "status": "in_progress",
        "depends_on": [],
    }]
    st["artifacts"] = [{"artifact_id": "a1", "step_id": "s1", "summary": "evidence"}]
    new_state, resp = d.apply(st, "complete_step", {"step_id": "s1"})
    assert resp.ok
    s1 = new_state["steps"][0]
    assert s1["round_status"] == "completed"
    assert s1["step_status"] == "completed"
    assert new_state["executing_step_id"] is None


def test_complete_step_steered_without_step_type_is_not_steered():
    """Fallback: a step whose entry lacks step_type (legacy seeding) must not
    silently inherit steered semantics — it mints step_status like an ordinary
    step, so pipeline advancement still terminates."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "survey"})
    st["executing_step_id"] = "s1"
    st["steps"] = [{"step_id": "s1", "status": "in_progress", "depends_on": []}]
    st["artifacts"] = [{"artifact_id": "a1", "step_id": "s1", "summary": "evidence"}]
    new_state, resp = d.apply(st, "complete_step", {"step_id": "s1"})
    assert resp.ok
    assert new_state["steps"][0]["step_status"] == "completed"
    assert new_state["steps"][0]["round_status"] == "completed"
    assert new_state["executing_step_id"] is None


def test_is_complete_requires_report_and_artifacts():
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "assemble_report"})
    assert d.is_complete(st) is False
    st["report_path"] = "/tmp/r.md"
    assert d.is_complete(st) is False
    st["artifacts"] = [{"artifact_id": "a1", "summary": "evidence"}]
    assert d.is_complete(st) is True


# ---------------------------------------------------------------------------
# Backtracker clue-chain carry (Stage 4)
# ---------------------------------------------------------------------------


def test_round_object_carries_recovery_journal():
    state = {
        "steps": [{"step_id": "s1", "status": "in_progress", "depends_on": []}],
        "artifacts": [],
        "executing_step_id": "s1",
        "phase": "deep_dive",
        "recovery": {"s1": {"clues": [{"text": "found X"}], "step_round": 1}},
    }
    obj = RoundObject.from_state(state)
    assert obj.recovery["s1"]["clues"][0]["text"] == "found X"


def test_backtrack_report_surfaces_clues():
    state = {
        "steps": [{"step_id": "s1", "status": "in_progress", "depends_on": []}],
        "artifacts": [],
        "executing_step_id": "s1",
        "phase": "deep_dive",
        "recovery": {"s1": {"clues": [{"text": "found X"}], "step_round": 1}},
    }
    report = Backtracker().rebuild(state, failed_step_id="s1")
    assert "found X" in report.context_rebuild
    assert "recovery clues" in report.context_rebuild


# ---------------------------------------------------------------------------
# Steering gate
# ---------------------------------------------------------------------------


def test_steering_gate_completes_on_artifact_count():
    from quro.domains.codebase_research.domain import _CodebaseResearchSteeringGate

    gate = _CodebaseResearchSteeringGate(min_artifacts=2)

    # No state → hint about no state.
    assert gate.suggested_sufficiency(None, objective="survey") == "No state available."

    # Empty artifacts → hint about zero artifacts.
    assert "0 artifact(s)" in gate.suggested_sufficiency({"artifacts": []}, objective="survey")

    # One artifact → hint about one artifact (min_artifacts=2).
    assert "1 artifact(s)" in gate.suggested_sufficiency(
        {"artifacts": [{"summary": "a"}]}, objective="survey"
    )

    # Two artifacts → hint about two artifacts.
    assert "2 artifact(s)" in gate.suggested_sufficiency(
        {"artifacts": [{"summary": "a"}, {"summary": "b"}]}, objective="survey"
    )

    # Keyword heuristic is gone — "synthesize" alone doesn't change the hint.
    assert "0 artifact(s)" in gate.suggested_sufficiency({"artifacts": []}, objective="synthesize")


def test_steering_gate_default_min_artifacts():
    from quro.domains.codebase_research.domain import _CodebaseResearchSteeringGate

    gate = _CodebaseResearchSteeringGate()
    # Default min_artifacts=1: one artifact hint.
    assert "1 artifact(s)" in gate.suggested_sufficiency(
        {"artifacts": [{"summary": "a"}]}, objective="anything"
    )


# ---------------------------------------------------------------------------
# carry_forward (fix-v20260906.md): structured dependency state handoff
# ---------------------------------------------------------------------------


def _dep_source():
    """A dependency source in the shape StepAdapter._apply_dependency_state builds."""
    return {
        "step_id": "survey-1",
        "keys": ["plan", "interpretation", "steps", "artifacts"],
        "steps": [
            {"step_id": "survey-1", "status": "completed", "step_status": "completed"},
        ],
        "artifacts": [
            {
                "artifact_id": "art_dep_1",
                "step_id": "survey-1",
                "kind": "analysis",
                "summary": "found module X",
                "body": "...",
            },
            {
                "artifact_id": "art_dep_2",
                "step_id": "survey-1",
                "kind": "analysis",
                "summary": "traced Y",
                "body": "...",
            },
        ],
    }


def test_carry_forward_allowed_in_every_phase():
    d = CodebaseResearchDomain()
    for phase in RESEARCH_PHASES:
        assert "carry_forward" in d.allowed_ops_for_phase(phase)


def test_carry_forward_merges_dependency_artifacts_and_steps():
    """Dependency artifacts land in state['artifacts'] tagged with the source
    step_id — the key that lets the hints block, steering_artifacts block, and
    ContextView.dependencies resolve them (fix-v20260906.md)."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "deep_dive"})
    st["executing_step_id"] = "deep-1"
    st, resp = d.apply(
        st, "carry_forward", {"sources": [_dep_source()], "phase": "step_execute"}
    )
    assert resp.ok
    pairs = [(a["artifact_id"], a["step_id"]) for a in st["artifacts"]]
    assert ("art_dep_1", "survey-1") in pairs
    assert ("art_dep_2", "survey-1") in pairs
    # Dependency step entries are merged by step_id.
    assert "survey-1" in [s["step_id"] for s in st["steps"]]


def test_carry_forward_dedupes_and_existing_entries_win():
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "deep_dive"})
    st["executing_step_id"] = "deep-1"
    st["steps"] = [{"step_id": "survey-1", "status": "in_progress"}]
    st["artifacts"] = [
        {"artifact_id": "art_dep_1", "step_id": "deep-1", "summary": "own artifact"}
    ]
    st, resp = d.apply(
        st, "carry_forward", {"sources": [_dep_source()], "phase": "step_execute"}
    )
    assert resp.ok
    # No duplicate ids; the existing entry keeps its own data.
    by_id = {a["artifact_id"]: a for a in st["artifacts"]}
    assert set(by_id) == {"art_dep_1", "art_dep_2"}
    assert by_id["art_dep_1"]["step_id"] == "deep-1"
    assert [s["step_id"] for s in st["steps"]] == ["survey-1"]


def test_carry_forward_does_not_overwrite_phase():
    """The adapter passes the engineering target phase ('step_execute'); a
    research session must keep its own step-derived phase or gating breaks."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "deep_dive"})
    st, resp = d.apply(
        st, "carry_forward", {"sources": [_dep_source()], "phase": "step_execute"}
    )
    assert resp.ok
    assert st["phase"] == "deep_dive"


def test_carry_forward_ignores_non_research_keys():
    """plan/interpretation are engineering state keys; the research op carries
    only steps + artifacts."""
    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "survey"})
    src = {
        "step_id": "s0",
        "keys": ["plan", "steps", "artifacts"],
        "plan": "some plan text",
        "steps": [],
        "artifacts": [],
    }
    st, resp = d.apply(st, "carry_forward", {"sources": [src]})
    assert resp.ok
    assert st.get("plan") is None
    assert st["artifacts"] == []


def test_carry_forward_hint_block_sees_dependency_artifacts():
    """End-to-end prompt effect (fix-v20260906.md): after carry_forward the
    hints block renders the dependency's artifacts for an access='hints' step."""
    from quro.context.blocks import render_block

    d = CodebaseResearchDomain()
    st = d.initial_state({"phase": "deep_dive"})
    st["executing_step_id"] = "deep-1"
    st, _ = d.apply(
        st, "carry_forward", {"sources": [_dep_source()], "phase": "step_execute"}
    )
    # The adapter seeds the current step entry after carry_forward.
    st["steps"].append({
        "step_id": "deep-1",
        "step_type": "deep_dive_symbol",
        "objective": "dive",
        "status": "in_progress",
        "depends_on": ["survey-1"],
        "access": "hints",
    })
    rendered = render_block("hints", st, phase="deep_dive", budget=4000, principal="worker")
    # access='hints' renders a compact table (bodies are pulled via tools), so
    # assert the dependency artifact rows are present with their source step.
    assert "art_dep_1" in rendered and "survey-1" in rendered
    assert "art_dep_2" in rendered


def test_adapter_carries_dependency_state_for_research_domain():
    """StepAdapter._apply_dependency_state must run for a research domain (the
    isinstance(EngineeringWorkflowDomain) gate is gone — fix-v20260906.md)."""
    from quro.runtime.adapter import StepAdapter
    from quro.steps.core import StepResult, StepSpec
    from quro_thinking.kernel import ReasoningSession

    domain = CodebaseResearchDomain()
    session = ReasoningSession(domain)
    init_resp = session.call(
        "codebase_research_init", {"problem": "p", "phase": "deep_dive"}
    )
    assert init_resp.ok

    dep_result = StepResult.success("survey-1", state={
        "phase": "survey",
        "steps": [{"step_id": "survey-1", "status": "completed", "step_status": "completed"}],
        "artifacts": [
            {"artifact_id": "art_dep_1", "step_id": "survey-1",
             "kind": "analysis", "summary": "found module X", "body": "..."},
        ],
    })

    step = StepSpec(id="deep-1", step_type="deep_dive_symbol", objective="dive",
                    depends_on=["survey-1"])
    adapter = StepAdapter(tool_registry=None, backend=None)
    adapter._apply_dependency_state(
        step, domain.step_types, session, {"results": {"survey-1": dep_result}}
    )

    st = session.state
    assert any(
        a.get("artifact_id") == "art_dep_1" and a.get("step_id") == "survey-1"
        for a in st["artifacts"]
    ), f"dependency artifact missing from state: {st['artifacts']}"
    # Session phase untouched by the handoff.
    assert st["phase"] == "deep_dive"
