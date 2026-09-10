"""ExecutionGovernor — the single policy coordinator for step lifecycle.

All step-lifecycle decisions (terminate / retry / rollback / feedback /
context) route through this component.  No other file decides "is this
step done?".

See ``docs/system-analysis/09-refactoring-implementation-plan.md`` §4.1.
"""

from __future__ import annotations

import enum
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from quro.core.resources import IResourceStore
from quro.core.domain.phase_ops import terminal_tool_for_phase
from quro.core.tools.coordinator import IToolCoordinator
from quro.logging import is_dump_context_enabled
from quro.context.coordinator import (
    NextInstructionBlockHook,
    ObjectiveBlockHook,
    ProjectScopeBlockHook,
    PromptCoordinator,
    SkillsBlockHook,
    StateBlocksHook,
    SystemBlocksHook,
)
from quro.core.domain.workflow_domain import EngineeringWorkflowDomain
from quro.context.default_session import DefaultSessionContext
from quro.context.models import SessionContextConfig
from quro.context.protocols import ISessionContext
from quro.runtime.base import CompletionResult, IRuntimeBackend
from quro.steps.core import StepResult, StepSpec
from quro.steps.hooks import HookContext
from quro_thinking.kernel import ReasoningSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal tools — the ONLY tools whose success can end a step with ok=True
# ---------------------------------------------------------------------------

TERMINAL_TOOLS: dict[str, str] = {
    "complete_step": "complete_step(step_id)",
    "finalize_step": "finalize_step()",
    "confirm_understanding": "confirm_understanding()",
    "submit_evaluation": "submit_evaluation(proposal)",
}


# ---------------------------------------------------------------------------
# TerminationReason — machine-readable exit cause
# ---------------------------------------------------------------------------


class TerminationReason(str, enum.Enum):
    TOOL_TERMINATED = "tool_terminated"  # terminal tool succeeded
    PHASE_DONE = "phase_done"            # domain state phase == "done"
    VERIFIED = "verified"                # explicit completion check passed
    IDLE_STALL = "idle_stall"            # no output progress for N rounds
    MAX_ROUNDS = "max_rounds"            # round budget exhausted
    EXCEPTION = "exception"              # uncaught error


@dataclass
class TerminationDecision:
    reason: TerminationReason
    ok: bool
    detail: str = ""
    attempts_used: int = 0
    underlying_error: Exception | None = None


# ---------------------------------------------------------------------------
# ProgressSnapshot — replaces the buggy idle counter
# ---------------------------------------------------------------------------


@dataclass
class ProgressSnapshot:
    """Captures observable progress within a step for stall detection.

    Unlike the old ``consecutive_idle_rounds`` counter that incremented
    unconditionally in non-``step_spec`` phases, this snapshot measures
    *actual* output progress across all phases.
    """

    phase: str = "understanding"
    step_count: int = 0
    artifact_count: int = 0
    terminal_calls: list[str] = field(default_factory=list)
    last_mutation_at: float = 0.0
    active_step_id: str | None = None
    # Cumulative number of tool invocations across rounds.  Read-only
    # exploration tools (ls/grep/read/MCP) never mutate domain state, so
    # without this counter a read-only step like the explore step is wrongly
    # declared
    # idle after a few rounds of actual work.
    tool_calls: int = 0

    def has_progress(self, prev: ProgressSnapshot) -> bool:
        """Return True when *any* observable progress occurred."""
        return (
            self.phase != prev.phase
            or self.step_count != prev.step_count
            or self.artifact_count != prev.artifact_count
            or bool(self.terminal_calls)
            or self.tool_calls != prev.tool_calls
        )

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> ProgressSnapshot:
        if state is None:
            return cls()
        steps = state.get("steps", [])
        return cls(
            phase=state.get("phase", "understanding"),
            step_count=len(steps),
            artifact_count=len(state.get("artifacts", [])),
            last_mutation_at=time.monotonic(),
            active_step_id=state.get("executing_step_id"),
        )


# ---------------------------------------------------------------------------
# ExecutionGovernor
# ---------------------------------------------------------------------------


class ExecutionGovernor:
    """Owns the step lifecycle: rounds, termination, prompt assembly.

    ``StepAdapter`` becomes a thin factory that assembles
    ``(step_type, session, tools, policies)`` and calls ``governor.execute()``.
    """

    def __init__(
        self,
        *,
        step: StepSpec,
        session: ReasoningSession,
        domain: EngineeringWorkflowDomain,
        coordinator: IToolCoordinator,
        principal: str = "worker",
        backend: IRuntimeBackend,
        max_rounds: int = 30,
        max_idle_rounds: int = 3,
        context_budget_chars: int = 12000,
        feedback_budget_chars: int = 3000,
        digest_events: int = 12,
        resource_store: IResourceStore | None = None,
        hook_context: HookContext | None = None,
        template_loader: Any = None,
        identity_block: str = "",
        model_name: str = "gpt-4o-mini",
        session_context: ISessionContext | None = None,
        grants: Any = None,
        project_scope: str = "",
        skills_block: str = "",
    ) -> None:
        self.step = step
        self.session = session
        self.domain = domain
        self.coordinator = coordinator
        self.principal = principal
        self.backend = backend
        self.max_rounds = max_rounds
        self.max_idle_rounds = max_idle_rounds
        self.context_budget_chars = context_budget_chars
        self.feedback_budget_chars = feedback_budget_chars
        self.digest_events = digest_events
        self.resource_store = resource_store
        self.hook_context = hook_context
        self.template_loader = template_loader
        self.identity_block = identity_block
        self.model_name = model_name
        self.session_context = session_context
        self.grants = grants
        self.project_scope = project_scope
        self.skills_block = skills_block

        # Lazy-imported policies (set via builder or defaults)
        self._termination_policy: TerminationPolicy | None = None
        self._prompt_policy: PromptAssemblyPolicy | None = None
        self._feedback: FailureFeedback | None = None

        # Steering message state: scoped to ONE round (a Steering Cycle is a
        # bounded L0 episode, Steering-Semantics.md §8.2/§9).  Each round
        # rebuilds _messages fresh; prior-round tool results cross the boundary
        # only as a compact digest (see execute_steering_round).
        self._messages: list[dict[str, Any]] = []
        self._steering_initialized: bool = False

    # -- Policy setters (used by StepAdapter to inject policies) ----------

    def set_termination_policy(self, policy: TerminationPolicy) -> None:
        self._termination_policy = policy

    def set_prompt_policy(self, policy: PromptAssemblyPolicy) -> None:
        self._prompt_policy = policy

    def set_feedback(self, feedback: FailureFeedback) -> None:
        self._feedback = feedback

    @property
    def termination_policy(self) -> TerminationPolicy:
        if self._termination_policy is None:
            from quro.governor.termination import DefaultTerminationPolicy
            self._termination_policy = DefaultTerminationPolicy(
                max_idle_rounds=self.max_idle_rounds,
            )
        return self._termination_policy

    @property
    def prompt_policy(self) -> PromptAssemblyPolicy:
        if self._prompt_policy is None:
            from quro.governor.prompt import DefaultPromptAssemblyPolicy
            self._prompt_policy = DefaultPromptAssemblyPolicy(
                context_budget_chars=self.context_budget_chars,
                template_loader=self.template_loader,
                role_name=self.step.step_type or None,
                fallback_system_prompt=self.identity_block,
            )
        return self._prompt_policy

    @property
    def feedback(self) -> FailureFeedback:
        if self._feedback is None:
            from quro.governor.feedback import FailureFeedback
            self._feedback = FailureFeedback(
                max_attempts_shown=5,
                budget_chars=self.feedback_budget_chars,
            )
        return self._feedback

    # ------------------------------------------------------------------
    # execute — the ONLY place that decides step completion
    # ------------------------------------------------------------------

    def execute(self) -> StepResult:
        """Run the step once, returning a ``StepResult``.

        A step runs a single attempt: on failure the result is handed
        back to the pipeline layer, which decides the next move
        (RecoveryHook salvage via the clue chain, or planner replan).
        There is deliberately **no in-governor retry/rollback** — rollback
        discarded already-committed clues and re-ran the step from
        scratch, which is exactly the "no awareness of prior progress"
        failure this removes.
        """
        # Governor-owned access log: audit-only.  The unified-resource-layer
        # phase 1 moves ACL enforcement up-front (the tool layer denies foreign
        # reads at call time), so this log is no longer a decision input — it
        # only records what was fetched this attempt.
        try:
            self.session._round_access_log = set()
        except (AttributeError, Exception):
            pass

        decision = self._run_attempt(attempt=0)
        return self._build_result(decision)

    def execute_steering_round(
        self,
        objective: str,
        *,
        next_instruction: Any = None,
    ) -> StepResult:
        """Execute one steering round as a fresh, bounded L0 episode.

        Called once per Steering Cycle.  Unlike ``execute()``:
        - Rebuilds a FRESH turn structure (``[system, user, ...]``) from the
          current state blocks every round — each Cycle is a bounded L0
          exploration episode with its own LLM context (Steering-Semantics.md
          §8.2, §9).
        - Carries prior-cycle tool results forward as a compact digest
          (evidence, ``_build_digest``), never as raw assistant/tool turns.
        - Runs a multi-turn inner loop until terminal tool or idle stall.
        - Returns the step result for this round only.

        The governor MUST be the same instance across steering rounds so the
        session state (artifacts, feedback, next instruction) and the prior
        transcript digest accumulate; the raw LLM turns do not.  Reusing raw
        assistant/tool turns across rounds produced incoherent transcripts:
        the old rebuild dropped the user pairing messages while keeping every
        assistant message, so stalled text-only turns accumulated into a run of
        consecutive assistant messages at the next round's start — llama-server
        rejected the call with "Cannot have 2 or more assistant messages at the
        end of the list".
        """
        state = self.session.state
        if state is None:
            return StepResult.failure(self.step.id, "No session state")

        phase = state.get("phase", "understanding")

        # Compact digest of the PRIOR round's tool results — cross-cycle
        # evidence (Steering-Semantics.md §4).  The current round's transcript
        # is rebuilt fresh below.
        prior_digest = self._build_digest(self._messages)

        # Assemble fresh system + user blocks from current state.  The round
        # OBJECTIVE is the LIVE narrowed objective (F3), not the frozen seed;
        # the seed rides as a context anchor in the ObjectiveBlockHook.
        ctx = self._get_ctx()
        hook_ctx = self.hook_context
        failure_digest = self.feedback.digest(self.session)
        system_blocks, user_blocks = self._assemble_prompt(
            state, phase, hook_ctx, objective=objective,
        )
        if failure_digest:
            user_blocks["feedback"] = failure_digest

        # Fresh turn structure per round (mirrors ``_run_rebuild``): a clean
        # ``[system, user, ...]`` head + the prior-round tool-result digest.
        # No assistant/tool message may cross the round boundary, so a round
        # can never begin with orphaned or consecutive assistant messages.
        self._messages = ctx.assemble_dicts(
            system_blocks=system_blocks,
            user_blocks=user_blocks,
        )
        self._assert_user_query(self._messages, phase, user_blocks)
        if prior_digest:
            self._messages.insert(1, {"role": "user", "content": prior_digest})

        # Print the next instruction for debugging visibility.
        ni_block = user_blocks.get("next_instruction", "")
        if ni_block:
            import sys
            print(f"  [step-agent] Next instruction: {ni_block[:300]}")
            sys.stdout.flush()

        # --- Inner round loop -------------------------------------------------------
        # The round executor is the step agent: it keeps the FULL step surface
        # (CONTRACT_TOOLS like add_artifact/complete_step plus the StepType's
        # filesystem/Shell grants) so it can explore, archive evidence, and
        # terminate.  The read-only one-shot steering act is already tool-less,
        # so there is no read-only surface left to protect here — restricting
        # this call stripped the executor of its terminal behavior (the F1
        # regression).  The executor respects the same ACL the non-steering
        # path enforces.
        #
        # Each steering round maps to ONE narrowed objective, so the round is a
        # short inner loop (F-R1): the step agent is given up to ``max_idle_rounds``
        # consecutive turns.  A text-only (no tool-call) turn is NOT an immediate
        # success — it nudges the agent toward terminal completion and, if it
        # recurs, ends the round as a failure (idle-stall detection, F-R2).
        # Terminal completion (an archived artifact + complete_step) ends the
        # round as a success.  Productive tool turns reset the idle counter and
        # keep the round alive so the agent can explore the current objective.
        tools = self.coordinator.surface(
            step=self.step, phase=phase, principal=self.principal,
        )
        idle_floor = self.max_idle_rounds or 0
        idle_count = 0
        max_turns = (idle_floor + 1) * 3  # hard cap: 3x idle budget
        turn_count = 0

        while idle_count <= idle_floor and turn_count < max_turns:
            turn_count += 1
            result = self.backend.complete_one(
                self._messages,
                tools=tools,
                log_prefix="[step-agent] ",
            )
            self._messages.append(self._format_assistant_message(result))

            if not result.tool_calls:
                # Text-only turn: not a success.  Nudge toward terminal completion
                # and count as an idle round.  Once the stall budget is exceeded
                # the round ends as a failure — never as a silent success.
                idle_count += 1
                if idle_count > idle_floor:
                    # Pair the assistant turn before returning to preserve the
                    # invariant: every assistant append must be followed by a
                    # user or tool message.  Without this, the next steering
                    # round inherits an orphaned assistant message, causing the
                    # LLM API to reject with "2 or more assistant messages".
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "[Round ended: idle stall exceeded. "
                            "No more turns will be taken.]"
                        ),
                    })
                    return StepResult.failure(
                        self.step.id,
                        "Steering round stalled: no tool calls for "
                        f"{idle_count} consecutive turns (no archived evidence, "
                        "no complete_step)",
                        state=state,
                    )
                self._messages.append({
                    "role": "user",
                    "content": (
                        "[No tool calls produced.  If the task is complete, "
                        "call add_artifact then complete_step.  Otherwise, take "
                        "the next action.]"
                    ),
                })
                continue

            for tc in result.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})

                # Defense-in-depth: if coordinator.invoke raises (e.g. corrupted
                # state from a prior bug), produce an error-shaped response so
                # the assistant message (appended above) is always paired with a
                # tool response.  Without this, the orphaned assistant message
                # triggers "2+ assistant messages" on the next complete_one call.
                try:
                    tool_response = self.coordinator.invoke(tool_name, tool_args)
                except Exception as exc:
                    tool_response = f"Error: {type(exc).__name__}: {exc}"
                    logger.warning(
                        "tool dispatch failed for '%s': %s", tool_name, exc
                    )

                if self._is_terminal_tool(tool_name) and not tool_response.startswith("Error"):
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{len(self._messages)}"),
                        "content": str(tool_response),
                    })
                    return self._build_result(TerminationDecision(
                        TerminationReason.TOOL_TERMINATED, ok=True,
                        detail=f"Terminal tool '{tool_name}' succeeded",
                        attempts_used=1,
                    ))

                if tool_name == "complete_step" and "DEP_ACCESS_REQUIRED" in tool_response:
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your complete_step was rejected — you must fetch "
                            "dependency artifacts first. Call get_artifact or "
                            "get_step, then add_artifact with citations, and "
                            "retry complete_step."
                        ),
                    })

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{len(self._messages)}"),
                    "content": str(tool_response),
                })

            # Productive turn — reset idle counter and continue the loop so
            # the step agent gets another turn to process tool results.
            idle_count = 0

        # Round's stall budget exhausted without terminal completion.
        # Defense-in-depth: if the last turn was text-only (no tool calls),
        # self._messages ends with a trailing assistant message.  Append a
        # synthetic user message to prevent "2 or more assistant messages"
        # errors on the next complete_one call.
        if self._messages and self._messages[-1].get("role") == "assistant":
            self._messages.append({
                "role": "user",
                "content": (
                    "[Round ended: max turns exceeded. "
                    "No more turns will be taken.]"
                ),
            })
        if turn_count >= max_turns:
            return StepResult.failure(
                self.step.id,
                f"Steering round exceeded max turns ({max_turns}): "
                "no terminal completion despite productive tool activity",
                state=state,
            )
        return StepResult.failure(
            self.step.id,
            "Steering round stalled: no terminal completion",
            state=state,
        )

    def _run_attempt(self, attempt: int) -> TerminationDecision:
        """One attempt: N rounds of (prompt → LLM → tool → check).

        Dispatches to ``_run_rebuild`` or ``_run_accumulate`` based on the
        resolved message strategy from ``HookContext.metadata``.
        """
        strategy = "rebuild"
        injection_interval = 2
        max_budget = 0
        if self.hook_context is not None:
            strategy = self.hook_context.metadata.get("message_strategy", "rebuild")
            injection_interval = self.hook_context.metadata.get(
                "message_state_injection_interval", 2
            )
            max_budget = self.hook_context.metadata.get("message_max_budget_chars", 0)

        print(f"  [{self._step_label()}] Reasoning — model={self.model_name}")
        sys.stdout.flush()

        if strategy == "accumulate":
            return self._run_accumulate(attempt, injection_interval, max_budget)
        else:
            return self._run_rebuild(attempt)

    def _run_rebuild(self, attempt: int) -> TerminationDecision:
        """REBUILD mode: reconstruct [system, user] each round with digest.

        Uses ``ISessionContext.assemble_dicts()`` for block-attributed
        message assembly with token estimation and trim-on-overflow.
        """
        ctx = self._get_ctx()

        round_index = 0
        prev_progress: ProgressSnapshot | None = None
        idle_count = 0
        total_tool_calls = 0
        terminal_tool_used: str | None = None

        # Use governor-level _messages so steering rounds can reuse context.
        if not self._steering_initialized:
            self._messages = []
            self._steering_initialized = True

        while round_index < self.max_rounds and self.session.state is not None:
            state = self.session.state
            phase = state.get("phase", "understanding")

            # --- Check termination -------------------------------------------------
            progress = ProgressSnapshot.from_state(state)
            progress.tool_calls = total_tool_calls
            if terminal_tool_used:
                progress.terminal_calls = [terminal_tool_used]

            # Phase-done check
            if phase == "done":
                return self._end_session(TerminationDecision(
                    TerminationReason.PHASE_DONE, ok=True,
                    detail="Domain phase reached 'done'",
                    attempts_used=attempt + 1,
                ))

            # Idle stall detection (progress-based, not counter-only)
            if prev_progress is not None and not progress.has_progress(prev_progress):
                idle_count += 1
            else:
                idle_count = 0
            if idle_count >= self.max_idle_rounds:
                return self._end_session(TerminationDecision(
                    TerminationReason.IDLE_STALL, ok=False,
                    detail=f"No output progress for {idle_count} consecutive rounds",
                    attempts_used=attempt + 1,
                ))
            prev_progress = progress

            # --- Assemble prompt via context layer ----------------------------------
            hook_ctx = self.hook_context

            # Failure feedback: inject prior failures and axioms so the model
            # does not repeat already-failed calls (§4.5 of the plan).
            failure_digest = self.feedback.digest(self.session)

            # Session digest (cross-round context)
            digest = self._build_digest(self._messages)

            system_blocks, user_blocks = self._assemble_prompt(state, phase, hook_ctx)
            if failure_digest:
                user_blocks["feedback"] = failure_digest

            self._messages = ctx.assemble_dicts(
                system_blocks=system_blocks,
                user_blocks=user_blocks,
            )
            self._assert_user_query(self._messages, phase, user_blocks)
            if digest:
                self._messages.insert(1, {"role": "user", "content": digest})

            # --- One LLM call = one round ------------------------------------------
            result = self.backend.complete_one(self._messages, tools=self.coordinator.surface(step=self.step, phase=phase, principal=self.principal))

            if not result.tool_calls:
                # Model returned text without tool calls — this counts as a
                # non-progress round; idle detection will catch it.
                # Prevent self._messages from ending with a trailing assistant
                # message when the loop exits (MAX_ROUNDS, IDLE_STALL).
                # Some LLM APIs (llama-server) reject "2 or more assistant
                # messages at the end of the list."  Append a synthetic user
                # nudge to keep the sequence valid.
                self._messages.append({
                    "role": "user",
                    "content": (
                        "[No tool calls produced.  If the task is complete, "
                        "call complete_step.  Otherwise, take the next action.]"
                    ),
                })
                round_index += 1
                continue

            total_tool_calls += len(result.tool_calls)
            self._messages.append(self._format_assistant_message(result))

            for tc in result.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})

                tool_response = self.coordinator.invoke(tool_name, tool_args)

                new_state = self.session.state
                new_phase = new_state.get("phase") if new_state else None

                # --- Terminal tool detection ---------------------------------------
                if self._is_terminal_tool(tool_name) and not tool_response.startswith("Error"):
                    # submit_evaluation: step ends regardless of proposal.
                    # ok is determined downstream (complete proposal → ok=True;
                    # needs_work → ok=False with replan_request).
                    return self._end_session(TerminationDecision(
                        TerminationReason.TOOL_TERMINATED, ok=True,
                        detail=f"Terminal tool '{tool_name}' succeeded",
                        attempts_used=attempt + 1,
                    ))

                # DEP_ACCESS_REQUIRED hint for complete_step
                if tool_name == "complete_step" and "DEP_ACCESS_REQUIRED" in tool_response:
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your complete_step was rejected — you must fetch "
                            "dependency artifacts first. Call get_artifact or "
                            "get_step, then add_artifact with citations, and "
                            "retry complete_step."
                        ),
                    })

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{len(self._messages)}"),
                    "content": str(tool_response),
                })

            round_index += 1

        # Round budget exhausted
        return self._end_session(TerminationDecision(
            TerminationReason.MAX_ROUNDS, ok=False,
            detail=f"Exhausted {self.max_rounds} rounds",
            attempts_used=attempt + 1,
        ))

    def _run_accumulate(
        self,
        attempt: int,
        injection_interval: int,
        max_budget: int,
    ) -> TerminationDecision:
        """ACCUMULATE mode: keep message history across rounds.

        Uses ``ISessionContext.assemble_dicts()`` for the initial
        block-attributed message assembly.  Subsequent rounds append to
        the message list directly.

        Messages accumulate until a terminal tool is called.  Fresh projected
        state is injected only on events (phase change, idle rounds), not
        every round.
        """
        ctx = self._get_ctx()

        state = self.session.state or {}
        phase = state.get("phase", "understanding")
        failure_digest = self.feedback.digest(self.session)

        system_blocks, user_blocks = self._assemble_prompt(
            state, phase, self.hook_context
        )
        if failure_digest:
            user_blocks["feedback"] = failure_digest

        messages: list[dict[str, Any]] = ctx.assemble_dicts(
            system_blocks=system_blocks,
            user_blocks=user_blocks,
        )
        self._assert_user_query(messages, phase, user_blocks)

        round_index = 0
        prev_progress: ProgressSnapshot | None = None
        idle_count = 0
        total_tool_calls = 0
        prev_phase = phase
        had_tool_calls = False

        while round_index < self.max_rounds and self.session.state is not None:
            state = self.session.state
            phase = state.get("phase", "understanding")

            progress = ProgressSnapshot.from_state(state)
            progress.tool_calls = total_tool_calls

            if phase == "done":
                return self._end_session(TerminationDecision(
                    TerminationReason.PHASE_DONE, ok=True,
                    detail="Domain phase reached 'done'",
                    attempts_used=attempt + 1,
                ))

            if prev_progress is not None and not progress.has_progress(prev_progress):
                idle_count += 1
            else:
                idle_count = 0
            if idle_count >= self.max_idle_rounds:
                return self._end_session(TerminationDecision(
                    TerminationReason.IDLE_STALL, ok=False,
                    detail=f"No output progress for {idle_count} consecutive rounds",
                    attempts_used=attempt + 1,
                ))
            prev_progress = progress

            # Event-driven state injection
            if prev_phase != phase:
                projected = self._render_user_view(state, phase)
                messages.append({
                    "role": "user",
                    "content": f"[Phase: {prev_phase} → {phase}]\n\n{projected}",
                })
                idle_count = 0
            elif not had_tool_calls:
                idle_rounds = sum(
                    1 for m in messages[-6:]
                    if m.get("role") == "user"
                    and "No tool calls" in str(m.get("content", ""))
                )
                if idle_rounds >= injection_interval:
                    projected = self._render_user_view(state, phase)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[No tool calls for {idle_rounds + 1} rounds. "
                            f"Updated state below.]\n\n{projected}"
                        ),
                    })

            if max_budget > 0:
                total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                if total_chars > max_budget:
                    self._trim_messages(messages, max_budget)

            result = self.backend.complete_one(messages, tools=self.coordinator.surface(step=self.step, phase=phase, principal=self.principal))

            if not result.tool_calls:
                messages.append(self._format_assistant_message(result))
                # Prevent two consecutive assistant messages at the end of the
                # list, which some LLM APIs (llama-server) reject with:
                #   "Cannot have 2 or more assistant messages at the end of the list."
                messages.append({
                    "role": "user",
                    "content": (
                        "[No tool calls produced.  If the task is complete, "
                        "call complete_step.  Otherwise, take the next action.]"
                    ),
                })
                round_index += 1
                had_tool_calls = False
                continue

            had_tool_calls = True
            total_tool_calls += len(result.tool_calls)
            messages.append(self._format_assistant_message(result))

            for tc in result.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})

                tool_response = self.coordinator.invoke(tool_name, tool_args)

                if self._is_terminal_tool(tool_name) and not tool_response.startswith("Error"):
                    return self._end_session(TerminationDecision(
                        TerminationReason.TOOL_TERMINATED, ok=True,
                        detail=f"Terminal tool '{tool_name}' succeeded",
                        attempts_used=attempt + 1,
                    ))

                if tool_name == "complete_step" and "DEP_ACCESS_REQUIRED" in tool_response:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your complete_step was rejected — you must fetch "
                            "dependency artifacts first. Call get_artifact or "
                            "get_step, then add_artifact with citations, and "
                            "retry complete_step."
                        ),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                    "content": str(tool_response),
                })

            round_index += 1
            prev_phase = phase

        return self._end_session(TerminationDecision(
            TerminationReason.MAX_ROUNDS, ok=False,
            detail=f"Exhausted {self.max_rounds} rounds",
            attempts_used=attempt + 1,
        ))

    def _trim_messages(
        self, messages: list[dict[str, Any]], max_budget: int
    ) -> None:
        """Trim oldest non-system messages when approaching budget."""
        keep_head = 2
        keep_tail = 8

        if len(messages) <= keep_head + keep_tail:
            return

        trimmed = (
            messages[:keep_head]
            + [{"role": "user", "content": "[Earlier interactions trimmed for brevity]"}]
            + messages[-keep_tail:]
        )
        messages[:] = trimmed

    # ------------------------------------------------------------------
    # Termination logging
    # ------------------------------------------------------------------

    _AGENT_TERMINATED = frozenset({TerminationReason.TOOL_TERMINATED})
    _SYSTEM_TERMINATED = frozenset({
        TerminationReason.PHASE_DONE,
        TerminationReason.IDLE_STALL,
        TerminationReason.MAX_ROUNDS,
    })

    def _is_terminal_tool(self, tool_name: str) -> bool:
        """Return True if *tool_name* ends the current step's session.

        Phase 0: termination is derived from the step's **StepType** (its
        ``executor_hints["terminal_tool"]``) with the current session phase as
        the fallback.  A step type that spans understanding → step_spec declares
        ``terminal_tool="finalize_step"`` so it continues past
        ``confirm_understanding``; a step type without a hint ends at its
        current phase's terminal tool.
        """
        if tool_name not in TERMINAL_TOOLS:
            return False
        hint_terminal = self._step_terminal_hint()
        if hint_terminal is not None:
            return tool_name == hint_terminal
        phase = (self.session.state or {}).get("phase", "understanding")
        phase_terminal = terminal_tool_for_phase(phase)
        if phase_terminal is not None:
            return tool_name == phase_terminal
        return True

    def _step_terminal_hint(self) -> str | None:
        """Return the step type's declared terminal tool, or None."""
        step_types = getattr(self.domain, "step_types", None)
        if step_types is None:
            return None
        hints = step_types.executor_hints_for(self.step.step_type)
        return hints.get("terminal_tool")

    def _step_label(self) -> str:
        """A short label for log lines: the step type, falling back to the id."""
        return self.step.step_type or self.step.id

    def _end_session(self, decision: TerminationDecision) -> TerminationDecision:
        """Log how the session ended: agent (terminal tool) or system."""
        label = self._step_label()
        if decision.reason in self._AGENT_TERMINATED:
            print(f"  [{label}] Session ended by agent: {decision.detail}")
        elif decision.reason in self._SYSTEM_TERMINATED:
            print(f"  [{label}] Session ended by system: {decision.reason.value}")
        else:
            print(f"  [{label}] Session ended: {decision.reason.value} — {decision.detail}")
        sys.stdout.flush()
        return decision

    def _assert_user_query(
        self,
        messages: list[dict[str, Any]],
        phase: str,
        user_blocks: dict[str, str],
    ) -> None:
        """Fail fast when no non-empty user message would reach the model.

        Some local models' chat templates raise ``No user query found in
        messages`` when the assembled list has no ``user`` message with
        content.  Print the stack and exit loudly so the offending
        phase/step is visible instead of an opaque upstream 500.
        """
        if any(m.get("role") == "user" and m.get("content") for m in messages):
            return
        import traceback
        traceback.print_stack()
        _step = getattr(self, "step", None)
        step_type = getattr(_step, "step_type", "") if _step is not None else ""
        raise RuntimeError(
            "No user query found in messages — "
            f"step_type={step_type!r} phase={phase!r} "
            f"user_blocks={list(user_blocks.keys())}"
        )

    def _assemble_prompt(
        self,
        state: dict[str, Any],
        phase: str,
        hook_ctx: HookContext | None,
        objective: str | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build (system_blocks, user_blocks) via the PromptCoordinator.

        ``objective`` overrides the OBJECTIVE block's live task.  In a
        steering round this is the *round* objective; the seed
        (``self.step.objective``) is demoted to a context anchor so the
        executor acts on the narrowed scope, not the frozen seed (F3).
        """
        system_prompt = self.prompt_policy.build_system(
            state=state, hook_context=hook_ctx,
        )
        step_label = self._step_label()
        live_objective = self.step.objective if objective is None else objective
        seed_anchor = "" if objective is None else self.step.objective
        hooks = [
            SystemBlocksHook(system_prompt),
            ProjectScopeBlockHook(self.project_scope),
            SkillsBlockHook(self.skills_block),
            ObjectiveBlockHook(self.step.id, live_objective, anchor=seed_anchor),
            NextInstructionBlockHook(
                getattr(self.step, "_next_instruction_ni", None)
                or self.step.next_instruction,
                state=state,
            ),
            StateBlocksHook(
                self._phase_block_names(phase),
                state, phase, self.context_budget_chars, step_label,
            ),
        ]
        if self.project_scope:
            logger.debug(
                "[governor] _assemble_prompt: project_scope hook registered (%d chars)",
                len(self.project_scope),
            )
        coordinator_kwargs: dict[str, Any] = {
            "principal": step_label,
            "domain": getattr(self.session, "domain", None),
            "phase": phase,
        }
        if self.grants is not None:
            from quro.core.resources import (
                DomainStateResolver,
                acl_for_state,
                principal_from_state,
            )
            coordinator_kwargs["grants"] = self.grants
            coordinator_kwargs["acl"] = acl_for_state(state)
            coordinator_kwargs["acl_principal"] = principal_from_state(state)
            coordinator_kwargs["resolver"] = DomainStateResolver(state)
        coordinator = PromptCoordinator(hooks, **coordinator_kwargs)
        system_blocks, user_blocks = coordinator.assemble()
        if is_dump_context_enabled():
            from quro.logging import dump_assembled_blocks
            dump = dump_assembled_blocks(
                blocks={**system_blocks, **user_blocks},
            )
            if dump:
                print(f"[governor] {dump}", file=sys.stderr)
        return system_blocks, user_blocks

    def _render_user_view(self, state: dict[str, Any], phase: str) -> str:
        """Render only the user view (for event-driven state injection)."""
        _, user_blocks = self._assemble_prompt(state, phase, self.hook_context)
        return "\n".join(user_blocks.values())

    def _phase_block_names(self, phase: str) -> list[str]:
        domain = getattr(self.session, "domain", None)
        mapping = getattr(domain, "PHASE_BLOCKS", None) if domain else None
        if not mapping:
            return []
        return list(mapping.get(phase, []))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        decision: TerminationDecision,
    ) -> StepResult:
        """Convert a termination decision into a StepResult."""
        final_state = self.session.state or {}
        artifacts = list(final_state.get("artifacts", []))

        if decision.ok:
            # Build event-log digest for debuggability
            event_log_digest: list[dict[str, Any]] = []
            try:
                log = getattr(self.session, "event_log", None)
                if log is not None and hasattr(log, "events"):
                    for ev in list(log.events)[-20:]:
                        event_log_digest.append({
                            "seq": getattr(ev, "seq", 0),
                            "op": getattr(ev, "op", ""),
                            "ok": getattr(ev, "ok", True),
                            "ts": str(getattr(ev, "timestamp", "")),
                        })
            except Exception:
                pass

            result = StepResult(
                step_id=self.step.id,
                ok=True,
                state=final_state,
                artifacts=artifacts,
                metrics={
                    "termination": decision.reason.value,
                    "attempts": decision.attempts_used,
                    "event_log_digest": event_log_digest,
                },
            )
            # Evaluator: ok only if a "complete" evaluation was submitted
            evaluations = final_state.get("evaluations", [])
            if decision.reason == TerminationReason.TOOL_TERMINATED and evaluations:
                if not any(e.get("proposal") == "complete" for e in evaluations):
                    result.ok = False
                    result.error = "Evaluator submitted evaluation but proposal is not 'complete'"
                    result.replan_request = {
                        "reason": "Evaluation requires further work",
                        "suggested_new_steps": [],
                        "suggested_removals": [],
                        "priority": "normal",
                    }
            return result

        # Failure path — no in-governor retry: hand control back up to the
        # pipeline layer.  The recovery hook (if attached) runs first and may
        # replace this with a recovered success; otherwise the planner hook
        # sees a high-priority replan request.
        result = StepResult(
            step_id=self.step.id,
            ok=False,
            state=final_state,
            artifacts=artifacts,
            error=decision.detail,
            metrics={
                "termination": decision.reason.value,
                "attempts": decision.attempts_used,
            },
        )
        result.replan_request = {
            "reason": decision.detail,
            "suggested_new_steps": [],
            "suggested_removals": [],
            "priority": "high",
        }

        return result

    def _build_digest(self, prior_messages: list[dict[str, Any]]) -> str:
        """Build a compact digest of recent tool results for cross-round context."""
        if not prior_messages:
            return ""
        tool_pairs: list[tuple[str, str]] = []
        i = len(prior_messages) - 1
        while i >= 0 and len(tool_pairs) < 6:
            msg = prior_messages[i]
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))[:200]
                tc_id = msg.get("tool_call_id", "")
                tool_name = "unknown"
                if i > 0:
                    prev = prior_messages[i - 1]
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == tc_id:
                                tool_name = tc.get("function", {}).get("name", "unknown")
                                break
                tool_pairs.append((tool_name, content))
            i -= 1
        if not tool_pairs:
            return ""
        tool_pairs.reverse()
        lines = ["--- Previous round tool results (for context) ---"]
        for name, content in tool_pairs:
            lines.append(f"[{name}]: {content}")
        return "\n".join(lines)

    def _format_assistant_message(self, result: CompletionResult) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for i, tc in enumerate(result.tool_calls)
            ]
        return msg

    def _get_ctx(self) -> DefaultSessionContext:
        """Return the session context, creating a default one if needed.

        Lazy-initialises a ``DefaultSessionContext`` with a token budget
        derived from ``context_budget_chars`` when no explicit
        ``session_context`` was provided.
        """
        if self.session_context is not None:
            if not isinstance(self.session_context, DefaultSessionContext):
                # Wrap external ISessionContext — caller is responsible
                # for lifecycle (open/close).
                return self.session_context  # type: ignore[return-value]
            return self.session_context

        # Auto-create a DefaultSessionContext.
        ctx = DefaultSessionContext()
        ctx.open_session(SessionContextConfig(
            max_tokens=self.context_budget_chars,
            compression_threshold=0.85,
        ))
        self.session_context = ctx
        return ctx


# ---------------------------------------------------------------------------
# Policy Protocols (imports deferred to avoid circular deps)
# ---------------------------------------------------------------------------


class TerminationPolicy(Protocol):
    def should_terminate(
        self,
        *,
        state: dict[str, Any],
        round_index: int,
        progress: ProgressSnapshot,
        last_decision: TerminationDecision | None,
    ) -> TerminationDecision | None:
        ...


class PromptAssemblyPolicy(Protocol):
    def build_system(
        self,
        *,
        state: dict[str, Any],
        hook_context: HookContext | None,
    ) -> str:
        ...


# Re-export for convenience
from quro.governor.termination import DefaultTerminationPolicy  # noqa: E402,F401
from quro.governor.prompt import DefaultPromptAssemblyPolicy  # noqa: E402,F401
from quro.governor.feedback import FailureFeedback  # noqa: E402,F401
