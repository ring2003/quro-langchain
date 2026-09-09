"""StepAdapter — bridges a ``StepSpec`` to quro-thinking's kernel reasoning loop.

Phase 0 cleanup: the former ``RoleAdapter`` is now step-driven.  A step's
identity is its ``StepType`` (identity block, skill pool, executor hints) —
there is no ``Role`` object.  Each ``execute()`` call creates a
``ReasoningSession`` backed by event sourcing, harness rules,
checkpoint/backtrack, and the full kernel safety rails.

Tool governance
---------------

All tool allocation and invocation route through a single ``ToolCoordinator``
(contract tools + lifecycle tools + ``StepType.tools`` grants, default-deny).
Internal (domain/framework) tools are always-on; their semantic availability
is enforced by the domain's own ``apply`` op table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from quro.core.resources import IResourceStore
from quro.core.domain.step_type import StepTypeCatalog
from quro.core.domain.workflow_domain import EngineeringWorkflowDomain
from quro.core.tools import (
    ToolAdapter,
    ToolCoordinator,
    domain_toolsets,
    framework_toolsets,
)
from quro.core.session.tools import make_skill_tools
from quro.runtime.base import CompletionResult, IRuntimeBackend
from quro.skills import render_skill_block
from quro.steps.checkpoint_hook import (
    REASONING_SESSION_ID_KEY,
    REASONING_STORAGE_DIR_KEY,
)
from quro.steps.core import StepExecutor, StepResult, StepSpec
from quro.steps.hooks import HookContext
from quro_thinking.kernel import ReasoningSession
from quro_thinking.storage import FileStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class StepAdapterConfig:
    """Runtime configuration for ``StepAdapter``.

    Args:
        max_rounds: Maximum reasoning rounds (LLM calls) per step.
        context_budget_chars: Character budget for state projection.
        state_dir: Directory for session/artifact persistence.
        auto_confirm: Skip model-switch confirmation prompts (for CI).
        default_model: Model used when the backend does not declare one.
        per_step_type_backends: Optional backend overrides keyed by step type
            (e.g. ``{"evaluate": evaluator_backend}``).  Falls back to the
            default backend when a step type has no override.
        default_message_strategy: Default message strategy when the StepSpec
            does not specify one.
        project_scope: Static project-scope context injected into every step's
            prompt.  Injected as a user-role ``project_scope`` block and
            prepended to the session-init problem.  Should tell the step agent
            *where* it is working (local filesystem paths, workspace root) and
            *how* to explore (use filesystem tools, not online search).
            Empty = no project scope block.
    """

    max_rounds: int = 12
    context_budget_chars: int = 12000
    state_dir: str = ".quro/pipeline"
    auto_confirm: bool = False
    default_model: str = "gpt-4o-mini"
    per_step_type_backends: dict[str, IRuntimeBackend] = field(default_factory=dict)
    default_message_strategy: str = "accumulate"
    project_scope: str = ""


class StepAdapter(StepExecutor):
    """Bridges a ``StepSpec`` to quro-thinking's kernel reasoning loop.

    Each ``execute()`` call creates a ``ReasoningSession`` backed by event
    sourcing, harness rules, checkpoint/backtrack, and the full kernel safety
    rails.  The adapter assembles a ``ToolCoordinator`` from the step's
    declarations and runs a ReAct loop that respects the harness discipline.

    Args:
        tool_registry: Registry of available file-system and MCP tools.
        backend: The LLM backend (implements ``IRuntimeBackend``).
        config: Runtime configuration for the reasoning loop.
        mcp_session: Optional MCP tool session for invoking external tools.
        domain_factory: Optional zero-arg factory for a user-authored domain
            (Phase D).  When None, the built-in engineering workflow domain is
            used.
        resource_store: Optional ``IResourceStore`` injected by the assembler
            (protocol-driven; the adapter never news a concrete backend).
            When None, artifacts stay in-memory (no persistence).
    """

    def __init__(
        self,
        tool_registry: Any,
        backend: IRuntimeBackend,
        *,
        config: StepAdapterConfig | None = None,
        mcp_session: Any = None,
        domain_factory: Any = None,
        resource_store: IResourceStore | None = None,
        resume_kernel_sessions: bool = False,
    ) -> None:
        self.tool_registry = tool_registry
        self.backend = backend
        self.config = config or StepAdapterConfig()
        self.mcp_session = mcp_session
        self._resource_store = resource_store
        # R0b (recovery-roadmap.md §3): when True, a step whose kernel
        # checkpoint snapshot exists is *resumed* (ReasoningSession.resume from
        # reasoning/{session_id}-{step_id}.checkpoint.json) instead of creating
        # an empty session.  Default False keeps the R0a fresh-session re-run.
        self.resume_kernel_sessions = resume_kernel_sessions
        self._per_step_type_backends: dict[str, IRuntimeBackend] = dict(
            self.config.per_step_type_backends
        )
        # Optional domain factory (Phase D).  When None, the built-in
        # engineering workflow domain is used.  A user-authored domain
        # (e.g. codebase research) is injected via this factory.
        self._domain_factory = domain_factory

    # ------------------------------------------------------------------
    # StepExecutor protocol
    # ------------------------------------------------------------------

    def execute(self, step: StepSpec, context: dict[str, Any]) -> StepResult:
        """Execute *step* within the given pipeline *context*.

        Args:
            step: The step specification (objective, step_type, tools, ...).
            context: Pipeline context with ``problem`` and ``results`` keys.

        Returns:
            ``StepResult`` with success/failure status, state, and artifacts.
        """
        step_types = self._step_types_catalog()

        # Select per-step-type backend when available, otherwise use default.
        backend = self._per_step_type_backends.get(step.step_type, self.backend)

        # Every step runs the domain — the planner is an outer ``g``,
        # not a step type (AGENTS.md "g/f separation").
        if self._domain_factory is not None:
            domain = self._domain_factory()
        else:
            domain = EngineeringWorkflowDomain()

        model_name = getattr(backend, "model_name", self.config.default_model)
        session = self._build_reasoning_session(domain, model_name, context)

        problem = self._build_problem(step, context)
        # R0b: a resumed kernel session is already initialised (state + event
        # log replayed) — re-init would be blocked by INIT_LOCKED, so skip it
        # and continue from the restored state.
        if not session.initialised:
            init_args: dict[str, Any] = {"problem": problem}
            # Phase D: derive the step's session phase from its step type when
            # the domain declares a phase vocabulary (phase gating per cognitive mode).
            phase_fn = getattr(domain, "phase_for_step_type", None)
            if callable(phase_fn):
                init_args["phase"] = phase_fn(step.step_type)
            init_resp = session.call(f"{domain.name}_init", init_args)
            if not init_resp.ok:
                return StepResult.failure(step.id, f"Init failed: {init_resp.error}")

        # --- State hand-off for dependent steps ---
        # Runs for any domain that declares a ``carry_forward`` op
        # (EngineeringWorkflowDomain, CodebaseResearchDomain).  Without it,
        # dependency artifacts never reach state["artifacts"] and the hints /
        # steering_artifacts blocks + ContextView.dependencies render empty —
        # only _build_problem's lossy DEPENDENCY ARTIFACTS text would carry
        # them (fix-v20260906.md).  A domain without the op gets a logged
        # warning from _apply_dependency_state; the step still executes.
        self._apply_dependency_state(step, step_types, session, context)

        # --- Ensure executing_step_id is set ---
        # In policy-loop mode the planner is an HTN solver (not an LLM step),
        # so no create_step is ever called.  We must seed the executor's
        # domain state with a step entry so complete_step can find it.
        _st = session.state
        if _st is not None:
            if not _st.get("executing_step_id"):
                _st["executing_step_id"] = step.id
            existing = {s["step_id"] for s in _st.get("steps", [])}
            if step.id not in existing:
                step_type = step_types.get(step.step_type)
                _st.setdefault("steps", []).append({
                    "step_id": step.id,
                    "step_type": step.step_type,
                    "objective": step.objective,
                    "status": "in_progress",
                    "depends_on": list(step.depends_on),
                    "access": step_type.access if step_type else "",
                })

        # Evaluate step types always start in the evaluate phase.
        phase_hint = step_types.executor_hints_for(step.step_type).get("phase", "")
        if phase_hint == "evaluate":
            st = session.state
            if st is not None and st.get("phase") == "understanding":
                st["phase"] = "evaluate"
                st["confirmed"] = True

        # --- Tool resolution ---
        # The store is injected by the assembler (protocol-driven); the
        # adapter never news a concrete backend.  ``resource_store`` may be
        # None, in which case artifacts stay in-memory.
        resource_store: IResourceStore | None = self._resource_store

        # Resolve the hook context before the coordinator so extra/remove tool
        # overrides ride the same governance path as declared tools.
        hook_ctx: HookContext | None = context.get("_hook_context")
        if hook_ctx is None:
            hook_ctx = HookContext(problem=problem)

        principal = self._principal(step)
        coordinator = self._build_coordinator(
            step, session, resource_store, domain, hook_ctx, step_types
        )

        # Resolve message strategy: StepSpec → StepAdapterConfig.
        strategy = step.message_strategy
        if strategy == "default":
            strategy = self.config.default_message_strategy
        hook_ctx.metadata["message_strategy"] = strategy
        hook_ctx.metadata["message_state_injection_interval"] = 2
        hook_ctx.metadata["message_max_budget_chars"] = 0

        # Phase B: render the SKILL block (declared → REQUIRED tips, rest of
        # the StepType pool → weak hints) into the step's system blocks.
        skill_block = render_skill_block(
            step.skills,
            domain.step_types.skill_pool_for(step.step_type),
            domain.skills,
        )

        # --- Delegate to ExecutionGovernor ---
        from quro.governor.core import ExecutionGovernor

        identity_block = step_types.identity_block_for(step.step_type)

        # Steering detection (inner g: objective override).  The executor's
        # identity stays EXECUTION-ONLY (the StepType identity_block above):
        # it does not own the objective-override loop, so no steering identity
        # is appended here (F2).  The loop-override role lives in the steering
        # act's own system prompt (Steering.system_prompt).
        steering_name = getattr(domain, "steering_for_step_type", {}).get(
            step.step_type, ""
        )

        governor = ExecutionGovernor(
            step=step,
            session=session,
            domain=domain,
            coordinator=coordinator,
            principal=principal,
            backend=backend,
            max_rounds=self.config.max_rounds,
            max_idle_rounds=3,
            context_budget_chars=self.config.context_budget_chars,
            resource_store=resource_store,
            hook_context=hook_ctx,
            identity_block=identity_block,
            model_name=model_name,
            grants=getattr(domain, "projection_grants", None),
            project_scope=self.config.project_scope,
            skills_block=skill_block,
        )
        if self.config.project_scope:
            print(
                f"  [adapter] project_scope wired: "
                f"{self.config.project_scope[:80]}..."
            )

        if steering_name:
            return self._execute_with_steering(
                step=step,
                context=context,
                session=session,
                domain=domain,
                step_types=step_types,
                backend=backend,
                governor=governor,
                steering_name=steering_name,
                coordinator=coordinator,
                resource_store=resource_store,
                hook_ctx=hook_ctx,
                model_name=model_name,
                problem=problem,
            )

        try:
            return governor.execute()
        except Exception as exc:
            logger.exception("StepAdapter.execute failed for step '%s'", step.id)
            return StepResult.failure(step.id, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_with_steering(
        self,
        *,
        step: StepSpec,
        context: dict[str, Any],
        session: ReasoningSession,
        domain: Any,
        step_types: StepTypeCatalog,
        backend: Any,
        governor: Any,
        steering_name: str,
        coordinator: Any,
        resource_store: Any,
        hook_ctx: Any,
        model_name: str,
        problem: str,
    ) -> StepResult:
        """Execute a step with SteeringLoop (Frontier re-calibration).

        The SteeringLoop drives progressive frontier re-calibration inside one
        composite step (Steering-Semantics.md §8–15).  Each cycle:
          1. Steering selects the next frontier (WHERE to investigate next)
          2. Steering creates a NextInstruction (focus = frontier representation)
          3. StepAgent/Governor executes with the frontier
          4. Evidence accumulates into understanding
          5. Steering re-calibrates (back to step 1)

        Semantic properties:
          - Seed Objective stays stable throughout
          - Frontier evolves after each cycle based on evidence
          - is_final_round is Steering's exit token, not StepAgent's
          - Compaction retries happen within a cycle (same frontier)

        The product is a single StepResult folded from all cycles.
        """
        from quro.runtime.steering import SteeringConfig, SteeringLoop, resolve_steering

        retry_budget = getattr(domain, "steering_retry_budget", 1)
        max_steering_rounds = getattr(domain, "steering_max_rounds", 8)

        # Domain must provide a gate and folder; default to always-complete
        # if the domain does not declare them (safe fallback for testing).
        gate = getattr(domain, "steering_gate", None)
        folder = getattr(domain, "steering_folder", None)

        # Fallback gate: budget cap only (domain-controlled).
        if gate is None:
            class _DefaultSteeringGate:
                def suggested_sufficiency(self, state: Any, *, objective: str) -> str:
                    return "Default gate: no sufficiency hint."
                def budget_exhausted(self, state: Any, *, rounds: int) -> bool:
                    return rounds >= max_steering_rounds
            gate = _DefaultSteeringGate()

        # Fallback folder: return last artifact.
        if folder is None:
            class _DefaultSteeringFolder:
                def fold(self, state: Any, *, artifacts: list[Any]) -> Any:
                    return artifacts[-1] if artifacts else None
            folder = _DefaultSteeringFolder()

        loop = SteeringLoop(
            act=resolve_steering(steering_name),
            gate=gate,
            folder=folder,
            config=SteeringConfig(
                max_rounds=max_steering_rounds,
                retry_budget=retry_budget,
            ),
        )

        # Build state_view from session for the steering act.
        # Include project_scope so the steering act knows WHERE the step is
        # working — without it, the act cannot make informed narrowing
        # decisions (adapter-prompt-view-violation.md).
        def _state_view() -> str:
            from quro.runtime.steering import build_context_view

            st = session.state or {}
            cv = build_context_view(st, step.id)
            lines: list[str] = []
            if self.config.project_scope:
                lines.append(f"Project scope:\n{self.config.project_scope}")
            phase = st.get("phase", "")
            if phase:
                lines.append(f"Phase: {phase}")
            # Render own_history (temporal)
            if cv.own_history:
                lines.append(f"\n## Own History ({len(cv.own_history)} prior rounds)")
                for a in cv.own_history:
                    lines.append(f"- [{a.get('artifact_id','')}] {a.get('summary','')[:120]}")
            # Render dependencies (spatial)
            if cv.dependencies:
                lines.append(f"\n## Dependency Artifacts ({len(cv.dependencies)})")
                for a in cv.dependencies:
                    lines.append(f"- [{a.get('artifact_id','')}] {a.get('summary','')[:120]}")
            # Render clues
            clues = st.get("recovery", {})
            for step_id, journal in clues.items():
                for c in journal.get("clues", []):
                    lines.append(f"Clue [{step_id}]: {c.get('text', '')[:120]}")
            return "\n".join(lines) or "(no progress yet)"

        # The act callable: executes the frontier for one steering cycle.
        def _act(
            frontier: str,
            *,
            signal: str = "",
            next_instruction: Any = None,
        ) -> tuple[Any, bool]:
            """Execute a steering cycle on the given frontier.

            The seed objective (step.objective) is NOT overridden — it stays
            stable across all cycles.  Only the frontier (current WHERE to
            investigate) changes; the frontier is delivered via the
            ``next_instruction`` block (NextInstruction.focus).

            Args:
                frontier: The selected frontier (WHERE to explore next).
                signal: Unused (legacy parameter).
                next_instruction: The NextInstruction carrying focus=frontier
                    and other steering metadata.

            Returns:
                (artifact, success) tuple where artifact is a StepResult.
            """
            # Update the session state's step entry with the frontier focus
            # for internal tracking (session state only).
            st = session.state
            if st is not None:
                for s in st.get("steps", []):
                    if s.get("step_id") == step.id:
                        s["objective"] = frontier  # Track current frontier
                        # Reset round_status for the new L0 session (fresh scope).
                        # This is NOT "resetting a lie" — it's correctly-scoped
                        # initialization for a fresh governor cycle.
                        s["round_status"] = "in_progress"
                        # Also reset step_status: for steered steps, step_status
                        # is the L1-owned token (only the fold may write
                        # "completed").  Resetting here prevents the premature
                        # step_status="completed" written by complete_step in a
                        # prior cycle from carrying over and causing the pipeline
                        # to advance before the gate approves completion.
                        s["step_status"] = "in_progress"
                        break

            # Store the structured NextInstruction for the governor's
            # NextInstructionBlockHook (carries the frontier representation).
            if next_instruction is not None:
                step._next_instruction_ni = next_instruction
            else:
                # Legacy path: clear any prior NextInstruction so the
                # governor falls back to step.next_instruction.
                if hasattr(step, "_next_instruction_ni"):
                    del step._next_instruction_ni

            try:
                result = governor.execute_steering_round(
                    frontier,
                    next_instruction=next_instruction,
                )
                return result, result.ok
            except Exception as exc:
                logger.warning(
                    "steering cycle failed for step '%s': %s", step.id, exc
                )
                return StepResult.failure(step.id, f"{type(exc).__name__}: {exc}"), False

        # Initialize steering loop parameters.
        # seed_objective stays stable across all cycles.
        seed_objective = step.objective
        step_objective = step.objective  # Anchor for Steering (stable)
        expected_output = step.expected_output

        drive_result = loop.drive(
            backend=backend,
            seed_objective=seed_objective,
            step_objective=step_objective,
            expected_output=expected_output,
            state_view=_state_view,  # callable — re-evaluated each cycle
            state=lambda: session.state,  # raw dict for the gate AND the act's
            # steering_artifacts block (full artifact content)
            act=_act,
        )
        rounds = drive_result.rounds
        exit_reason = drive_result.exit_reason  # "final_round" | "budget_exhausted"

        # Fold: collect all cycle artifacts, merge into one StepResult.
        # The last successful result is the terminal one.
        last_result: StepResult | None = None
        all_artifacts: list[dict[str, Any]] = []
        for r in rounds:
            if r.artifact is not None and isinstance(r.artifact, StepResult):
                last_result = r.artifact
                all_artifacts.extend(r.artifact.artifacts or [])
            elif isinstance(r.artifact, dict):
                all_artifacts.append(r.artifact)

        # Control-Ownership Enforcement (Steering-Semantics.md §12–15):
        # - is_final_round is Steering's exit token (§13): the steering act
        #   declares it from evidence; the loop exits on it (journaled).
        # - complete_step() is L0's exit token (ends current governor session);
        #   round_status is an L0-owned FACT for journaling/diagnostics only —
        #   it must never gate pipeline advancement (Anti-pattern E).
        # - For steered steps, step_status is minted by the fold (L1-owned),
        #   only when the loop exited on a Steering-approved terminal.
        #
        # Mint condition is the drive() exit reason, not L0-owned round_status:
        #   - exit_reason == "final_round" → the act declared the investigation
        #     complete: mint step_status="completed".
        #   - exit_reason == "budget_exhausted" → domain's call: mint only if
        #     the step produced artifacts; otherwise report failure.
        #
        # _act resets step_status and round_status each cycle, so the fold's
        # mint is the only writer of a steered step's step_status.
        st = session.state or {}
        step_artifacts = [
            a for a in st.get("artifacts", []) if a.get("step_id") == step.id
        ]
        steering_approved = (
            exit_reason == "final_round"
            or (exit_reason == "budget_exhausted" and bool(step_artifacts))
        )

        # L1-owned token mint: for steered steps, complete_step (L0) only sets
        # round_status — step_status remains "in_progress" (reset by _act each
        # round).  Only a Steering-approved exit mints step_status here.
        # For ordinary steps this is a no-op (complete_step already minted
        # step_status directly, L0 == L1).
        if steering_approved:
            for s in st.get("steps", []):
                if s.get("step_id") == step.id:
                    s["step_status"] = "completed"
                    break

        if last_result is not None:
            # Pipeline-advancement key: step_status (not round_status).
            step_done = any(
                s.get("step_id") == step.id and s.get("step_status") == "completed"
                for s in st.get("steps", [])
            )
            completed = steering_approved and step_done
            if not completed:
                error = (
                    f"Steering loop ended without a Steering-approved terminal "
                    f"(exit_reason={exit_reason!r}, rounds={len(rounds)}, "
                    f"artifacts={len(step_artifacts)})"
                )
            else:
                error = last_result.error
            ok = last_result.ok and completed
            # Merge artifacts from all cycles into the last result.
            merged = StepResult(
                step_id=step.id,
                ok=ok,
                state=last_result.state,
                artifacts=all_artifacts,
                error=error,
                metrics={
                    **(last_result.metrics or {}),
                    "steering_cycles": len(rounds),
                    "steering_frontiers": [r.objective for r in rounds],  # Frontier history
                    "steering_completed": steering_approved,
                    "steering_exit_reason": exit_reason,
                },
            )
            return merged

        return StepResult.failure(
            step.id,
            f"Steering loop produced no results after {len(rounds)} rounds",
        )

    def _build_reasoning_session(
        self,
        domain: Any,
        model_name: str,
        context: dict[str, Any],
    ) -> ReasoningSession:
        """Create the kernel reasoning session, honoring the checkpoint hook.

        The ``CheckpointHook`` hands the stable kernel session id + reasoning
        storage dir over via ``hook_ctx.metadata`` (reference, never copy).
        When present the kernel session is reused across resume and its event
        log is persisted under ``reasoning/``; without a hook the session keeps
        the kernel's default random id and stays in-memory.

        R0b (``resume_kernel_sessions=True``): when a persisted checkpoint
        snapshot exists for the stable session id, the session is *resumed* via
        ``ReasoningSession.resume`` (state + event log replayed) instead of
        creating an empty one.  A missing/malformed snapshot falls back to a
        fresh session — soft, never fatal (recovery-roadmap.md §2.3).
        """
        hook_ctx = context.get("_hook_context")
        reasoning_session_id: str | None = None
        reasoning_storage = None
        if hook_ctx is not None:
            reasoning_session_id = hook_ctx.metadata.get(REASONING_SESSION_ID_KEY)
            storage_dir = hook_ctx.metadata.get(REASONING_STORAGE_DIR_KEY)
            if storage_dir:
                try:
                    reasoning_storage = FileStorageBackend(storage_dir)
                except Exception:
                    logger.debug("Could not create reasoning storage at %s", storage_dir)

        if (
            self.resume_kernel_sessions
            and reasoning_storage is not None
            and reasoning_session_id is not None
        ):
            snapshot = reasoning_storage.load_snapshot(reasoning_session_id)
            if snapshot is not None:
                try:
                    return ReasoningSession.resume(
                        domain,
                        snapshot,
                        storage=reasoning_storage,
                    )
                except Exception:
                    logger.debug(
                        "kernel session resume failed for '%s'; falling back to fresh",
                        reasoning_session_id,
                        exc_info=True,
                    )

        return ReasoningSession(
            domain,
            session_id=reasoning_session_id,
            model_name=model_name,
            storage=reasoning_storage,
        )

    def _step_types_catalog(self) -> StepTypeCatalog:
        """Return the domain's StepType catalog (from factory or built-in)."""
        if self._domain_factory is not None:
            return self._domain_factory().step_types
        return EngineeringWorkflowDomain().step_types

    def _apply_dependency_state(
        self,
        step: StepSpec,
        step_types: StepTypeCatalog,
        session: ReasoningSession,
        context: dict[str, Any],
    ) -> None:
        """Carry forward state from dependency results via ``carry_forward`` domain op."""
        if not step.depends_on:
            return

        state_after_init = session.state
        if state_after_init is None:
            return

        results = context.get("results", {})

        # Determine target phase and which keys to transfer, from the step's
        # StepType phase hint.
        phase = step_types.executor_hints_for(step.step_type).get("phase", "")
        if phase == "step_spec":
            target_phase = "step_spec"
            keys = ["plan", "interpretation"]
        elif phase == "evaluate":
            target_phase = "evaluate"
            keys = ["plan", "steps", "artifacts", "evaluations"]
        else:
            target_phase = "step_execute"
            keys = ["plan", "interpretation", "steps", "artifacts"]

        # Build sources from dependency results.
        sources: list[dict[str, Any]] = []
        for dep_id in step.depends_on:
            dep_result = results.get(dep_id)
            if dep_result is None or not dep_result.state:
                continue
            src: dict[str, Any] = {"step_id": dep_id, "keys": keys}
            for key in keys:
                val = dep_result.state.get(key)
                if val:
                    src[key] = val
            sources.append(src)

        if not sources:
            return

        resp = session.call("carry_forward", {
            "sources": sources,
            "phase": target_phase,
        })
        if not resp.ok:
            logger.warning("carry_forward failed for step '%s': %s", step.id, resp.error)

    def _build_problem(self, step: StepSpec, context: dict[str, Any]) -> str:
        parts: list[str] = []

        # Inject project scope so the step agent knows where it is working.
        # This is adapter-provided operational context — it reaches the prompt
        # view via ProjectScopeBlockHook AND here for session-init context
        # (adapter-prompt-view-violation.md §4.3, Option A).
        if self.config.project_scope:
            parts.append(self.config.project_scope)
            logger.debug(
                "[adapter] _build_problem: project_scope injected (%d chars)",
                len(self.config.project_scope),
            )

        # A step sees a self-contained task (objective + expected output +
        # validation).  There is deliberately NO fallback that injects the raw
        # problem into a worker without an objective — an executor without an
        # objective is rejected earlier.
        parts.append(f"OBJECTIVE: {step.objective}")
        if step.expected_output:
            parts.append(f"EXPECTED OUTPUT: {step.expected_output}")
        if step.validation:
            parts.append(f"VALIDATION: {step.validation}")

        results = context.get("results", {})
        lines: list[str] = []
        for dep_id in step.depends_on:
            dep_result = results.get(dep_id)
            if dep_result is None:
                continue
            dep_state = dep_result.state or {}
            dep_plan = dep_state.get("plan", "")
            dep_interp = dep_state.get("interpretation", "")
            if dep_plan:
                lines.append(f"[{dep_id}] PLAN FROM PLANNER:\n{dep_plan}")
            if dep_interp:
                lines.append(f"[{dep_id}] Interpretation: {dep_interp}")
            for artifact in dep_result.artifacts:
                summary = artifact.get("summary", "")
                body = artifact.get("body", "")
                entry = f"[{dep_id}] {summary}" if summary else f"[{dep_id}] {artifact.get('artifact_id', '')}"
                if body:
                    entry = f"{entry}\n    {body[:200]}"
                lines.append(entry)

        if lines:
            parts.append("DEPENDENCY ARTIFACTS:\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def _build_coordinator(
        self,
        step: StepSpec,
        session: ReasoningSession,
        resource_store: IResourceStore | None,
        domain: Any,
        hook_ctx: HookContext,
        step_types: Any | None = None,
    ) -> ToolCoordinator:
        """Assemble the ``ToolCoordinator`` for one step.

        Internal (domain/framework) tools come from the toolsets and are
        always-on; external (filesystem) grants come from the registry and
        are default-deny (``StepType.tools``); MCP tools are server-level and
        always-on; skill tools and hook additions are always-on extras.
        """
        toolsets = framework_toolsets(resource_store) + domain_toolsets(domain)
        internal_tools = {t.name: t for t in ToolAdapter(session).build(toolsets)}

        # External filesystem grants by name (default-deny via StepType.tools).
        external_tools = {t.name: t for t in self.tool_registry.worker_tools}

        # MCP tools: server-level, always-on (ACL ownership is deferred).
        if self.mcp_session is not None:
            mcp_tools = list(self.mcp_session.get_tools())
        else:
            mcp_tools = list(self.tool_registry.mcp_tools)

        # Skill tools + hook additions are always-on extras.
        extra_tools = make_skill_tools(domain.skills) + list(hook_ctx.extra_tools)

        return ToolCoordinator(
            internal_tools=internal_tools,
            external_tools=external_tools,
            mcp_tools=mcp_tools,
            extra_tools=extra_tools,
            remove_tools=set(hook_ctx.remove_tools),
            mcp_session=self.mcp_session,
            step_types=step_types,
        )

    @staticmethod
    def _principal(step: StepSpec) -> str:
        """Return the lifecycle principal for *step*."""
        if step.step_type == "evaluate":
            return "evaluator"
        return "worker"

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
