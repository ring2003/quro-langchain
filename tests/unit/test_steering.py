"""Tests for the Steering tier (primitive-step.md §3.2/§4/§7).

Covers:
- the steering act emits a NextInstruction (structured) or str (legacy);
- the loop drives objective override and stops on complete / budget / is_final_round;
- the retry_budget is the time-disjoint boundary value (declared, not guessed);
- SteeringRegistry per-name prompt registration;
- NextInstruction parsing and rendering.
"""

from __future__ import annotations

from quro.runtime.base import FakeBackend
from quro.runtime.steering import (
    NextInstruction,
    Steering,
    SteeringConfig,
    SteeringLoop,
    SteeringRegistry,
    SteeringRound,
    _DEFAULT_STEERING_SYSTEM_PROMPT,
    _parse_next_instruction,
    default_steering_registry,
    resolve_steering,
)


class _AlwaysCompleteGate:
    def suggested_sufficiency(self, state, *, objective: str) -> str:
        return "Always complete: sufficiency hint."
    def verify_completion(self, state, *, objective: str) -> bool:
        return True
    def budget_exhausted(self, state, *, rounds: int) -> bool:
        return False


class _CountGate:
    def __init__(self, complete_at: int) -> None:
        self._complete_at = complete_at

    def suggested_sufficiency(self, state, *, objective: str) -> str:
        if objective == f"objective-{self._complete_at}":
            return f"Reached objective-{self._complete_at}, sufficiency reached."
        return f"Current objective: {objective}, not yet complete."

    def verify_completion(self, state, *, objective: str) -> bool:
        return objective == f"objective-{self._complete_at}"

    def budget_exhausted(self, state, *, rounds: int) -> bool:
        return False


class _NightFolder:
    def fold(self, state, *, artifacts: list) -> list:
        return artifacts


# ---------------------------------------------------------------------------
# NextInstruction parsing
# ---------------------------------------------------------------------------


def test_parse_structured_output():
    raw = (
        "FOCUS: Read the auth module\n"
        "SCOPE_IN: auth.py, token_utils.py\n"
        "SCOPE_OUT: tests/\n"
        "RECALL: auth_design_doc\n"
        "DONE_WHEN: token flow documented\n"
        "FINAL: no"
    )
    ni = _parse_next_instruction(raw, round_index=1)
    assert isinstance(ni, NextInstruction)
    assert ni.focus == "Read the auth module"
    assert ni.scope_in == ["auth.py", "token_utils.py"]
    assert ni.scope_out == ["tests/"]
    assert ni.recall == ["auth_design_doc"]
    assert ni.done_when == "token flow documented"
    assert ni.is_final_round is False
    assert ni.round_index == 1


def test_parse_final_round():
    raw = "FOCUS: done\nFINAL: yes"
    ni = _parse_next_instruction(raw, round_index=2)
    assert ni.is_final_round is True


def test_parse_plain_text_fallback():
    raw = "survey the parser module"
    ni = _parse_next_instruction(raw, round_index=0)
    assert ni.focus == "survey the parser module"
    assert ni.is_final_round is False


def test_parse_archived_token_in_plain_text():
    raw = "STEP_OBJECTIVE_ARCHIVED"
    ni = _parse_next_instruction(raw, round_index=0)
    assert ni.is_final_round is True


def test_next_instruction_objective_alias():
    ni = NextInstruction(round_index=0, focus="do stuff")
    assert ni.objective == "do stuff"


# ---------------------------------------------------------------------------
# SteeringRegistry
# ---------------------------------------------------------------------------


def test_steering_registry_register_and_resolve():
    reg = SteeringRegistry()
    reg.register("narrow", system_prompt="custom sys", user_prompt="custom usr {round_index}")
    assert reg.has("narrow")
    assert reg.names() == ["narrow"]
    prompts = reg.get_prompts("narrow")
    assert prompts == ("custom sys", "custom usr {round_index}")


def test_steering_registry_unknown_returns_none():
    reg = SteeringRegistry()
    assert reg.get_prompts("unknown") is None


def test_resolve_steering_with_registry():
    reg = SteeringRegistry()
    reg.register("narrow", system_prompt="NARROW_SYS")
    act = resolve_steering("narrow", registry=reg)
    assert isinstance(act, Steering)
    assert act.name == "narrow"
    assert act.system_prompt == "NARROW_SYS"


def test_resolve_steering_fallback_to_default():
    act = resolve_steering("unknown_name")
    assert isinstance(act, Steering)
    assert act.name == "unknown_name"
    # Should have the default prompt.
    assert "steering act" in act.system_prompt


def test_default_steering_registry_singleton():
    r1 = default_steering_registry()
    r2 = default_steering_registry()
    assert r1 is r2


# ---------------------------------------------------------------------------
# Steering act
# ---------------------------------------------------------------------------


def test_steering_act_emits_next_instruction():
    backend = FakeBackend(responses=["  survey the parser module  "])
    act = Steering()
    result = act.next_objective(
        state_view="seen: src/quro/parser",
        objective="overview the repo",
        step_objective="understand the codebase",
        expected_output="module list",
        prior_objectives=[],
        round_index=0,
        backend=backend,
    )
    assert isinstance(result, NextInstruction)
    assert result.focus == "survey the parser module"
    assert result.round_index == 0


def test_steering_act_requires_backend():
    from quro.runtime.steering import SteeringError

    act = Steering()
    try:
        act.next_objective(
            state_view="x", objective="y",
            step_objective="orig", expected_output="out",
            prior_objectives=[], round_index=0, backend=None,
        )
    except SteeringError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SteeringError for missing backend")


def test_steering_act_sees_full_artifact_bodies_from_state():
    """The L1 fix: the act's prompt carries FULL artifact content (own +
    dependencies), not just a count/summary index — so it steers toward gaps
    in artifact CONTENT, not around an opaque count."""
    from tests.unit.test_prompt_blocks import _steering_artifact_state

    act = Steering()
    prompt = act.render_user_prompt(
        state_view="seen",
        objective="obj",
        step_objective="seed",
        expected_output="out",
        prior_objectives=[],
        round_index=0,
        state=_steering_artifact_state(),
    )
    assert "art-a" in prompt
    assert "OWN BODY CONTENT" in prompt
    assert "art-b" in prompt
    assert "DEP BODY CONTENT" in prompt


def test_steering_act_prompt_empty_artifact_detail_without_state():
    """Without raw state the artifact view is empty (direct unit-test calls)."""
    act = Steering()
    prompt = act.render_user_prompt(
        state_view="seen",
        objective="obj",
        step_objective="seed",
        expected_output="out",
        prior_objectives=[],
        round_index=0,
    )
    assert "Artifacts produced so far" in prompt
    assert "art-" not in prompt


# ---------------------------------------------------------------------------
# SteeringLoop
# ---------------------------------------------------------------------------


def _act_compat(objective, *, signal="", next_instruction=None):
    return ({"summary": objective}, True)


def test_loop_stops_on_complete():
    """Test that the loop stops when is_final_round is set in the response."""
    # The act returns raw strings that parse to the desired NextInstructions.
    backend = FakeBackend(responses=[
        "FOCUS: objective-1\nFINAL: yes",
    ])
    loop = SteeringLoop(
        act=Steering(),
        gate=_AlwaysCompleteGate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=10, retry_budget=3),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_act_compat,
    )
    assert len(rounds) == 1
    assert rounds[0].objective == "objective-1"


def test_loop_overrides_objective_each_round():
    """Test that the loop overrides the objective each round and stops on is_final_round."""
    responses = [
        "FOCUS: objective-1",
        "FOCUS: objective-2",
        "FOCUS: objective-3\nFINAL: yes",
    ]
    backend = FakeBackend(responses=responses)
    loop = SteeringLoop(
        act=Steering(),
        gate=_CountGate(complete_at=3),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=3, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="objective-0",
        step_objective="objective-0",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_act_compat,
    )
    assert [r.objective for r in rounds] == [
        "objective-1",
        "objective-2",
        "objective-3",
    ]
    assert isinstance(rounds[0], SteeringRound)
    assert rounds[0].success is True


def test_loop_override_role_lives_in_act_prompt():
    """The loop-override role lives in the steering act's prompt, not the executor.

    F2: the executor's identity stays execution-only; the "objective is the
    seed of an exploration loop / you may override it" clause belongs to the
    steering act (Steering.system_prompt), where the act actually drives the
    override loop.
    """
    text = _DEFAULT_STEERING_SYSTEM_PROMPT
    assert "steering act" in text
    assert "seed of an exploration loop" in text
    assert "override" in text


def test_loop_calls_state_view_each_round():
    """Verify state_view is re-evaluated every round, not a stale snapshot."""
    call_count = 0
    views = []

    def _state_view() -> str:
        nonlocal call_count
        call_count += 1
        views.append(f"view-{call_count}")
        return f"view-{call_count}"

    backend = FakeBackend(responses=[
        "FOCUS: obj-1",
        "FOCUS: obj-2",
        "FOCUS: obj-3\nFINAL: yes",
    ])
    act = Steering()
    round_views: list[str] = []

    def _tracked_act(objective, *, signal="", next_instruction=None):
        round_views.append(views[-1] if views else "none")
        return ({"summary": objective}, True)

    loop = SteeringLoop(
        act=act,
        gate=_CountGate(complete_at=3),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=3, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=_state_view,
        state=lambda: {},
        act=_tracked_act,
    )
    assert len(rounds) == 3
    # state_view was called 3 times (once per round), not once.
    assert call_count == 3
    # Each round saw a different view.
    assert round_views == ["view-1", "view-2", "view-3"]


def test_loop_passes_state_to_gate():
    """Verify gate receives the raw state dict, not None."""
    received_states: list = []

    class _StateRecordingGate:
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            received_states.append(state)
            return "State recorded."
        def verify_completion(self, state, *, objective: str) -> bool:
            return len(received_states) >= 1  # Complete after 1 round
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    session_state = {"artifacts": [{"summary": "found stuff"}], "phase": "survey"}
    backend = FakeBackend(responses=[
        "FOCUS: obj-1",
    ])
    loop = SteeringLoop(
        act=Steering(),
        gate=_StateRecordingGate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: session_state,
        act=_act_compat,
    )
    assert len(rounds) == 1
    # Gate received the actual state dict, not None.
    assert received_states == [session_state]


def test_loop_detects_archived_token_and_forwards_signal():
    """STEP_OBJECTIVE_ARCHIVED token sets is_final_round but does NOT exit.

    With the fact layer, FINAL: yes / STEP_OBJECTIVE_ARCHIVED are
    journalized but not used as exit signals.  The loop continues until
    verify_completion returns True or budget exhausted.
    """
    # Round 1: normal objective. Round 2: termination token.
    backend = FakeBackend(responses=["obj-1", "STEP_OBJECTIVE_ARCHIVED"])
    received_signals: list[str] = []

    def _tracking_act(objective, *, signal="", next_instruction=None):
        received_signals.append(signal)
        return ({"summary": objective}, True)

    # Gate completes after 2 rounds (via verify_completion, not archived token).
    class _RoundGate:
        def __init__(self, complete_after: int) -> None:
            self._complete_after = complete_after
            self._rounds = 0

        def suggested_sufficiency(self, state, *, objective: str) -> str:
            self._rounds += 1
            if self._rounds >= self._complete_after:
                return "Sufficiency reached."
            return "Not yet complete."

        def verify_completion(self, state, *, objective: str) -> bool:
            return self._rounds >= self._complete_after

        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    loop = SteeringLoop(
        act=Steering(),
        gate=_RoundGate(complete_after=2),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_tracking_act,
    )
    # Round 1: no signal (first round).
    assert received_signals[0] == ""
    # Round 2: signal is now empty (STEP_OBJECTIVE_ARCHIVED no longer forwarded).
    assert received_signals[1] == ""
    # Loop ran 2 rounds, exited via verify_completion.
    assert len(rounds) == 2


def test_loop_stops_on_is_final_round():
    """FINAL: yes is journalized but does NOT exit the loop anymore.

    With the fact layer, verify_completion() is the exit authority.
    FINAL: yes is stored in the round's next_instruction for audit
    but the loop continues until verify_completion returns True.
    """
    backend = FakeBackend(responses=[
        "FOCUS: obj-1",
        "FOCUS: obj-2\nFINAL: yes",
    ])
    received_ni: list = []

    def _tracking_act(objective, *, signal="", next_instruction=None):
        received_ni.append(next_instruction)
        return ({"summary": objective}, True)

    # Gate completes after 2 rounds via verify_completion.
    class _CompleteAfter2Gate:
        def __init__(self):
            self._rounds = 0
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            return "Never complete: no sufficiency hint."
        def verify_completion(self, state, *, objective: str) -> bool:
            self._rounds += 1
            return self._rounds >= 2
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    loop = SteeringLoop(
        act=Steering(),
        gate=_CompleteAfter2Gate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_tracking_act,
    )
    # Loop ran both rounds, exited via verify_completion.
    assert len(rounds) == 2
    # Round 1's act received the NextInstruction from round 1.
    assert received_ni[0] is not None
    assert isinstance(received_ni[0], NextInstruction)
    assert received_ni[0].is_final_round is False
    # Round 2's act received the NextInstruction from round 1 (forwarded).
    assert received_ni[1] is not None
    assert isinstance(received_ni[1], NextInstruction)
    assert received_ni[1].is_final_round is False
    # The round itself recorded FINAL: yes via next_instruction.
    assert rounds[1].next_instruction is not None
    assert rounds[1].next_instruction.is_final_round is True


def test_loop_carries_next_instruction_in_rounds():
    """SteeringRound stores the NextInstruction when available."""
    backend = FakeBackend(responses=[
        "FOCUS: step-1\nSCOPE_IN: a.py",
        "FOCUS: step-2\nFINAL: yes",
    ])

    class _CompleteAfter2Gate:
        def __init__(self):
            self._rounds = 0
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            return "Never complete: no sufficiency hint."
        def verify_completion(self, state, *, objective: str) -> bool:
            self._rounds += 1
            return self._rounds >= 2
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    loop = SteeringLoop(
        act=Steering(),
        gate=_CompleteAfter2Gate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_act_compat,
    )
    assert len(rounds) == 2
    assert rounds[0].next_instruction is not None
    assert rounds[0].next_instruction.scope_in == ["a.py"]
    assert rounds[1].next_instruction is not None
    assert rounds[1].next_instruction.is_final_round is True


def test_round_executor_surface_is_full_step_surface():
    """F1 regression: the steering ROUND runs with the step's FULL surface.

    The round executor (execute_steering_round) is the step's execution body —
    it must keep add_artifact / complete_step / terminate_subtree plus the
    StepType's filesystem (Readonly) and Shell grants.  The read-only one-shot
    steering act is tool-less, so no read-only strip may land on the round.
    """
    from quro.core.tools import (
        ToolAdapter,
        ToolCoordinator,
        domain_toolsets,
        framework_toolsets,
    )
    from quro.core.worker_tools import all_worker_tools
    from quro.domains.codebase_research import CodebaseResearchDomain
    from quro.steps.core import StepSpec

    class _OkResp:
        ok = True

    class _FakeSession:
        def call(self, op, args):
            return _OkResp()

    domain = CodebaseResearchDomain()
    step = StepSpec(
        id="r1_survey",
        name="r1_survey",
        step_type="survey_module",  # declared in steering_for_step_type
        objective="survey the repo",
        tools=(),
    )
    internal = {
        t.name: t
        for t in ToolAdapter(_FakeSession()).build(
            framework_toolsets() + domain_toolsets(domain)
        )
    }
    external = {t.name: t for t in all_worker_tools()}
    coordinator = ToolCoordinator(
        internal_tools=internal,
        external_tools=external,
        extra_tools=[],
        step_types=domain.step_types,
    )
    surface = coordinator.surface(step=step, phase="survey", principal="worker")
    names = {t.name for t in surface}

    # The round executor must be able to explore, archive, and terminate.
    assert {"add_artifact", "complete_step", "terminate_subtree"} <= names
    # StepType grants (survey_module → Readonly | Shell) must be present.
    assert {"read", "ls", "grep", "shell"} <= names


def test_steering_round_rebuilds_fresh_context_per_round():
    """Each steering round is a fresh L0 episode: no raw turns cross the boundary.

    A Steering Cycle runs a bounded L0 exploration episode with its own LLM
    context (Steering-Semantics.md §8.2, §9).  The old cross-round accumulation
    carried every assistant/tool turn into the next round but dropped the user
    pairing messages, so stalled text-only turns produced a run of consecutive
    assistant messages at the next round's start (llama-server 400 "Cannot have
    2 or more assistant messages at the end of the list").
    """
    from unittest.mock import MagicMock
    from quro.governor.core import ExecutionGovernor
    from quro.steps.core import StepResult, StepSpec

    mock_session = MagicMock()
    mock_session.state = {"phase": "step_execute", "steps": [], "artifacts": []}
    mock_session.initialised = True
    mock_session._round_access_log = set()
    mock_session.event_log = MagicMock()
    mock_session.event_log.events = []

    mock_domain = MagicMock()
    mock_domain.name = "test_domain"
    mock_domain.step_types = MagicMock()
    mock_domain.step_types.executor_hints_for.return_value = {}
    mock_domain.skills = []

    mock_coordinator = MagicMock()
    mock_coordinator.surface.return_value = []
    mock_coordinator.invoke.return_value = "tool result"

    mock_backend = MagicMock()
    mock_backend.model_name = "test-model"
    # Round 1 (step_execute): read_file → complete_step (terminal for step_execute)
    # Round 2 (understanding): read_file → confirm_understanding (terminal for understanding)
    mock_backend.complete_one.side_effect = [
        MagicMock(content="", tool_calls=[{"id": "call_1", "name": "read_file", "args": {"path": "a.py"}}]),
        MagicMock(content="done", tool_calls=[{"id": "call_2", "name": "complete_step", "args": {}}]),
        MagicMock(content="", tool_calls=[{"id": "call_3", "name": "read_file", "args": {"path": "b.py"}}]),
        MagicMock(content="done", tool_calls=[{"id": "call_4", "name": "confirm_understanding", "args": {}}]),
    ]

    mock_resource_store = MagicMock()
    mock_hook_ctx = MagicMock()
    mock_hook_ctx.metadata = {"message_strategy": "rebuild"}
    mock_hook_ctx.extra_tools = []
    mock_hook_ctx.remove_tools = []

    step = StepSpec(
        id="step-1", name="explore", step_type="explore",
        objective="explore codebase", tools=(),
    )

    governor = ExecutionGovernor(
        step=step,
        session=mock_session,
        domain=mock_domain,
        coordinator=mock_coordinator,
        backend=mock_backend,
        max_rounds=10,
        resource_store=mock_resource_store,
        hook_context=mock_hook_ctx,
        identity_block="test identity",
    )

    # Round 1
    result1 = governor.execute_steering_round("objective-1")
    assert result1.ok

    # Round 2
    mock_session.state = {
        "phase": "understanding", "steps": [], "artifacts": [{"summary": "found"}]
    }
    result2 = governor.execute_steering_round("objective-2")
    assert result2.ok

    # Round 2 rebuilt its context from scratch: the transcript contains only
    # its OWN assistant/tool turns (no raw prior-round turns carried over),
    # and prior-cycle tool results travel forward as a compact digest only.
    assistant_msgs = [m for m in governor._messages if m.get("role") == "assistant"]
    tool_msgs = [m for m in governor._messages if m.get("role") == "tool"]
    assert len(assistant_msgs) <= 2, "round 2 must not inherit prior-round assistant turns"
    assert len(tool_msgs) <= 2, "round 2 must not inherit prior-round tool turns"
    assert any(
        m.get("role") == "user" and "Previous round tool results" in str(m.get("content"))
        for m in governor._messages
    ), "prior-cycle tool results must be carried as a digest (evidence)"

    # Fresh head: system first, then user blocks (an evidence digest may follow).
    assert governor._messages[0]["role"] == "system"
    assert any(m.get("role") == "user" for m in governor._messages[1:])


def test_steering_round_stall_does_not_poison_next_round():
    """Idle-stall text turns must not be carried into the next round's context.

    Regression for the llama-server 400 ("Cannot have 2 or more assistant
    messages at the end of the list"): round 1 stalls with consecutive text-only
    turns; round 2 must rebuild a fresh, valid turn structure instead of
    inheriting the stalled assistant tail.
    """
    from unittest.mock import MagicMock

    governor, mock_backend = _governor_for_steering_round()
    # phase step_execute makes complete_step terminal for round 2.
    governor.session.state = {"phase": "step_execute", "steps": [], "artifacts": []}
    # Round 1: four text-only turns → idle stall → failure.
    # Round 2: terminal complete_step → success.
    mock_backend.complete_one.side_effect = [
        MagicMock(content="stalled", tool_calls=[]),
        MagicMock(content="stalled", tool_calls=[]),
        MagicMock(content="stalled", tool_calls=[]),
        MagicMock(content="stalled", tool_calls=[]),
        MagicMock(content="done", tool_calls=[
            {"id": "call_5", "name": "complete_step", "args": {}},
        ]),
    ]

    r1 = governor.execute_steering_round("objective-1")
    assert r1.ok is False
    assert "stalled" in (r1.error or "")

    r2 = governor.execute_steering_round("objective-2")
    assert r2.ok is True

    # The round-2 transcript starts from a fresh, valid turn structure:
    # system → user... — no carried-over assistant tail, no run of consecutive
    # assistant messages (the llama-server 400 condition).
    roles = [m.get("role") for m in governor._messages]
    assert roles[0] == "system"
    for i in range(len(roles) - 1):
        assert not (roles[i] == "assistant" and roles[i + 1] == "assistant"), (
            "consecutive assistant messages at round 2 — llama-server 400 condition"
        )


def _governor_for_steering_round(max_idle_rounds=3):
    """Build an ExecutionGovernor rigged for a steering round via MagicMock."""
    from unittest.mock import MagicMock
    from quro.governor.core import ExecutionGovernor
    from quro.steps.core import StepSpec

    mock_session = MagicMock()
    mock_session.state = {"phase": "understanding", "steps": [], "artifacts": []}
    mock_session.initialised = True
    mock_session._round_access_log = set()
    mock_session.event_log = MagicMock()
    mock_session.event_log.events = []

    mock_domain = MagicMock()
    mock_domain.name = "test_domain"
    mock_domain.step_types = MagicMock()
    mock_domain.step_types.executor_hints_for.return_value = {}
    mock_domain.skills = []

    mock_coordinator = MagicMock()
    mock_coordinator.surface.return_value = []
    mock_coordinator.invoke.return_value = "tool result"

    mock_backend = MagicMock()
    mock_backend.model_name = "test-model"

    mock_resource_store = MagicMock()
    mock_hook_ctx = MagicMock()
    mock_hook_ctx.metadata = {"message_strategy": "rebuild"}
    mock_hook_ctx.extra_tools = []
    mock_hook_ctx.remove_tools = []

    step = StepSpec(
        id="step-1", name="explore", step_type="survey_module",
        objective="explore codebase", tools=(),
    )

    governor = ExecutionGovernor(
        step=step,
        session=mock_session,
        domain=mock_domain,
        coordinator=mock_coordinator,
        backend=mock_backend,
        max_rounds=10,
        max_idle_rounds=max_idle_rounds,
        resource_store=mock_resource_store,
        hook_context=mock_hook_ctx,
        identity_block="test identity",
    )
    return governor, mock_backend


def test_steering_round_text_only_is_not_silent_success():
    """    F-R1: a text-only steering round (no tool calls) is NOT ok=True.

    The auto-end bug: a round whose model outputs prose was rewarded with
    StepResult(ok=True) immediately, so the step could "finish" without ever
    archiving evidence or calling complete_step.  It must be nudged toward
    terminal completion and, after the stall budget, end as a FAILURE.
    """
    from unittest.mock import MagicMock

    governor, mock_backend = _governor_for_steering_round(max_idle_rounds=2)
    mock_backend.complete_one.return_value = MagicMock(
        content="some prose, no tools", tool_calls=[]
    )

    result = governor.execute_steering_round("objective-1")

    assert result.ok is False
    assert "stalled" in (result.error or "")
    # The nudge must have been injected (one per text-only turn).
    nudge_msgs = [
        m for m in governor._messages
        if m.get("role") == "user" and "No tool calls produced" in str(m.get("content"))
    ]
    assert len(nudge_msgs) >= 1


def test_steering_round_terminal_tool_succeeds():
    """F-R1: a round that calls a terminal tool (complete_step) ends as success.

    Terminal completion is the only path that yields ok=True from a steering
    round — completing the step with evidence.
    """
    from unittest.mock import MagicMock

    governor, mock_backend = _governor_for_steering_round()
    # Override session phase to step_execute where complete_step IS terminal.
    governor.session.state = {"phase": "step_execute", "steps": [], "artifacts": []}
    # complete_step is a terminal tool for step_execute phase.
    mock_backend.complete_one.return_value = MagicMock(
        content="completing",
        tool_calls=[{"id": "call_1", "name": "complete_step", "args": {}}],
    )

    result = governor.execute_steering_round("objective-1")

    assert result.ok is True


def test_steering_round_productive_tools_continues_loop():
    """F-R1: a round with productive (non-terminal) tool calls continues the loop.

    Productive exploration resets the idle counter and gives the step agent
    another turn to process tool results.  The round only succeeds when a
    terminal tool is called; non-terminal activity alone is not success.
    """
    from unittest.mock import MagicMock

    governor, mock_backend = _governor_for_steering_round()
    # Override session phase to step_execute where complete_step IS terminal.
    governor.session.state = {"phase": "step_execute", "steps": [], "artifacts": []}
    # First call: non-terminal tool, second call: terminal tool
    mock_backend.complete_one.side_effect = [
        MagicMock(content="exploring", tool_calls=[{"id": "call_1", "name": "read_file", "args": {"path": "a.py"}}]),
        MagicMock(content="done", tool_calls=[{"id": "call_2", "name": "complete_step", "args": {}}]),
    ]

    result = governor.execute_steering_round("objective-1")

    assert result.ok is True
    # Verify the loop ran two turns (non-terminal then terminal)
    assert mock_backend.complete_one.call_count == 2


def test_final_yes_does_not_exit_when_verify_completion_false():
    """FINAL: yes no longer exits the loop; only verify_completion does.

    The old behavior: FINAL: yes → is_final_round → loop breaks.
    The new behavior: FINAL: yes is journalized but ignored for exit.
    The loop exits only when gate.verify_completion() returns True.
    """
    # Gate never completes (verify_completion always False).
    class _NeverCompleteGate:
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            return "Never complete."
        def verify_completion(self, state, *, objective: str) -> bool:
            return False
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    # Act returns FINAL: yes on round 1 — should NOT exit the loop.
    # Backend cycles through responses, so round 2+ reuse round 1's response.
    backend = FakeBackend(responses=[
        "FOCUS: obj-1\nFINAL: yes",
    ])
    round_count = 0

    def _counting_act(objective, *, signal="", next_instruction=None):
        nonlocal round_count
        round_count += 1
        return ({"summary": objective}, True)

    loop = SteeringLoop(
        act=Steering(),
        gate=_NeverCompleteGate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=3, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_counting_act,
    )
    # The loop ran 3 rounds despite FINAL: yes on every round (cycled response).
    assert len(rounds) == 3
    assert round_count == 3


def test_verify_completion_true_exits_loop():
    """When gate.verify_completion() returns True, the loop exits."""
    class _CompleteAfter1Gate:
        def __init__(self):
            self._rounds = 0
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            return "hint"
        def verify_completion(self, state, *, objective: str) -> bool:
            self._rounds += 1
            return self._rounds >= 2  # Complete after 2 rounds
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    backend = FakeBackend(responses=[
        "FOCUS: obj-1",
        "FOCUS: obj-2",
        "FOCUS: obj-3",
    ])

    loop = SteeringLoop(
        act=Steering(),
        gate=_CompleteAfter1Gate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "",
        state=lambda: {},
        act=_act_compat,
    )
    # Exited after 2 rounds when verify_completion returned True.
    assert len(rounds) == 2


def test_steering_uses_coordinator_hooks_when_provided():
    """When coordinator_hooks is provided, Steering uses PromptCoordinator."""
    from quro.context.coordinator import ObjectiveBlockHook

    # Use _sequence for complete_one path.
    backend = FakeBackend(sequence=["FOCUS: obj-1\nFINAL: yes"])

    # Provide a coordinator hook.
    hooks = [ObjectiveBlockHook("step-1", "test objective")]
    act = Steering(name="test", coordinator_hooks=hooks)

    result = act.next_objective(
        state_view="progress so far",
        objective="current obj",
        step_objective="original obj",
        expected_output="output",
        prior_objectives=[],
        round_index=0,
        backend=backend,
    )
    assert isinstance(result, NextInstruction)
    assert result.focus == "obj-1"


def test_steering_falls_back_to_legacy_without_hooks():
    """Without coordinator_hooks, Steering uses the legacy hand-built prompt."""
    backend = FakeBackend(responses=["FOCUS: legacy-obj"])

    act = Steering(name="legacy")
    result = act.next_objective(
        state_view="state",
        objective="obj",
        step_objective="step-obj",
        expected_output="out",
        prior_objectives=[],
        round_index=0,
        backend=backend,
    )
    assert isinstance(result, NextInstruction)
    assert result.focus == "legacy-obj"


def test_steering_loop_records_journal_events():
    """SteeringLoop.drive records SteeringRoundEntry events to the journal."""
    from quro.planner.session_journal import InMemoryMetaPlannerSessionJournal

    journal = InMemoryMetaPlannerSessionJournal()

    class _CompleteAfter2Gate:
        def __init__(self):
            self._rounds = 0
        def suggested_sufficiency(self, state, *, objective: str) -> str:
            return "hint"
        def verify_completion(self, state, *, objective: str) -> bool:
            self._rounds += 1
            return self._rounds >= 2
        def budget_exhausted(self, state, *, rounds: int) -> bool:
            return False

    backend = FakeBackend(responses=[
        "FOCUS: obj-1",
        "FOCUS: obj-2",
    ])

    loop = SteeringLoop(
        act=Steering(),
        gate=_CompleteAfter2Gate(),
        folder=_NightFolder(),
        config=SteeringConfig(max_rounds=5, retry_budget=1),
    )
    rounds = loop.drive(
        backend=backend,
        seed_objective="seed",
        step_objective="seed",
        expected_output="",
        state_view=lambda: "view",
        state=lambda: {},
        act=_act_compat,
        journal=journal,
    )
    assert len(rounds) == 2

    # Verify journal recorded events.
    history = journal.steering_history()
    # Should have 2 round entries + 1 exit_reason entry = 3 total
    assert len(history) >= 2
    # First entry should be for round 0.
    assert history[0].round_index == 0
    assert history[0].objective == "obj-1"
    # Should have an exit_reason entry.
    exit_entries = [e for e in history if e.exit_reason]
    assert len(exit_entries) >= 1
    assert exit_entries[0].exit_reason == "verify_completion"
