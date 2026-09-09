"""Recovery hook — passive salvage when a step ends with no artifact.

Blueprint §6: a post-step hook fires **when a step ends with no artifact**
("a step with no artifact has no surviving value").  It re-runs the step in a
fresh, clean session seeded with the step's recovery journal
(``step_round N``, ``clues[]``, ``continue_reasons[]``, ``failed_reasons[]``),
so the recovery model can freely ``complete_step`` (assemble the artifact from
clues), ``continue_step`` (resume exploration from the clues), or give up.

The hook is runtime scaffolding — it is **not** the domain solver.  It holds a
``StepExecutor`` (the same executor the pipeline uses) and re-executes the step
with the recovery journal injected as the user prompt.  Convergence is soft:
``max_recovery_rounds`` bounds the number of re-runs (the prompt strongly
steers give-up past the threshold); the policy-loop stall/replan is the hard
backstop (blueprint §6 Convergence).
"""

from __future__ import annotations

import logging
from typing import Any

from quro.core.features import Feature
from quro.steps.core import StepResult, StepSpec
from quro.steps.hooks import HookContext, HookFactoryContext

logger = logging.getLogger(__name__)


class RecoveryHook:
    """Re-run an artifact-less step seeded with its recovery journal.

    Enabled per step-type by the runtime from the domain's recovery
    declaration (decision 12).  Constructed with the pipeline's step executor
    so a re-run reuses the exact same domain / tool / backend wiring.
    """

    name: str = "recovery"

    def __init__(
        self,
        step_executor: Any,
        *,
        max_recovery_rounds: int = 3,
    ) -> None:
        self._step_executor = step_executor
        self._max_recovery_rounds = max(1, max_recovery_rounds)

    # -- StepHook interface -------------------------------------------------

    def on_pre_step(self, step: StepSpec, context: HookContext) -> StepSpec | None:
        """No-op: recovery only acts after the step runs."""
        return None

    def on_post_step(
        self,
        step: StepSpec,
        result: StepResult,
        context: HookContext,
    ) -> StepResult | None:
        """Recover a step that ended unsuccessfully.

        Returns a replacement ``StepResult`` when recovery ran, or ``None``
        when the step already succeeded / is not recovery-enabled.
        """
        if Feature.RECOVERY.value not in step.features:
            return None
        if result.ok:
            return None

        state = result.state or {}
        journal = (state.get("recovery") or {}).get(step.id, {})
        clues = list(journal.get("clues", []))
        round_n = int(journal.get("step_round", 0)) + 1

        logger.info(
            "Recovery hook: step '%s' ended unsuccessfully (step_type=%s, "
            "clues=%d, round=%d)",
            step.id, step.step_type, len(clues), round_n,
        )

        recovered: StepResult = result
        while round_n <= self._max_recovery_rounds:
            recovered = self._run_recovery(step, context, round_n, clues, journal)
            if recovered.ok:
                logger.info("Recovery hook: step '%s' recovered (ok)", step.id)
                return recovered

            rs = recovered.state or {}
            journal = (rs.get("recovery") or {}).get(step.id, {})
            clues = list(journal.get("clues", []))
            if not clues:
                # No clues to resume from — no surviving value to continue on.
                logger.info("Recovery hook: step '%s' has no clues to resume from", step.id)
                break
            round_n = int(journal.get("step_round", 0)) + 1

        logger.warning(
            "Recovery hook: step '%s' exhausted recovery rounds — giving up "
            "(policy loop backstops)",
            step.id,
        )
        return recovered

    # -- internals ----------------------------------------------------------

    def _run_recovery(
        self,
        step: StepSpec,
        context: HookContext,
        round_n: int,
        clues: list[dict[str, Any]],
        journal: dict[str, Any],
    ) -> StepResult:
        prompt = _build_recovery_prompt(
            step, round_n, clues, journal, self._max_recovery_rounds
        )
        recovery_ctx = HookContext(
            problem=context.problem,
            step_results=dict(context.step_results),
            user_prompt=prompt,
        )
        recovery_ctx.metadata = dict(context.metadata)
        pipe_ctx = {
            "problem": context.problem,
            "results": dict(context.step_results),
            "_hook_context": recovery_ctx,
        }
        try:
            return self._step_executor.execute(step, pipe_ctx)
        except Exception as exc:  # noqa: BLE001 - recovery must not crash the pipeline
            logger.exception("Recovery hook: re-run of step '%s' raised", step.id)
            return StepResult.failure(
                step.id, f"recovery re-run failed: {type(exc).__name__}: {exc}"
            )


def _step_artifacts(step_id: str, result: StepResult) -> list[dict[str, Any]]:
    """Return the artifacts produced by *step_id* in *result*."""
    state = result.state or {}
    return [a for a in state.get("artifacts", []) if a.get("step_id") == step_id]


def _build_recovery_prompt(
    step: StepSpec,
    round_n: int,
    clues: list[dict[str, Any]],
    journal: dict[str, Any],
    max_recovery_rounds: int,
) -> str:
    lines: list[str] = []
    lines.append("## STEP RECOVERY")
    lines.append(
        f"Your previous attempt at step '{step.id}' ended without producing an "
        "artifact, so the work has no surviving value."
    )
    lines.append(f"This is recovery round {round_n} of {max_recovery_rounds}.")
    lines.append("")
    lines.append(f"Objective: {step.objective}")
    lines.append("")

    lines.append(f"Committed clues ({len(clues)}):")
    for c in clues:
        lines.append(f"- {c.get('text', '')}")
    lines.append("")

    continue_reasons = journal.get("continue_reasons", [])
    failed_reasons = journal.get("failed_reasons", [])
    if continue_reasons:
        lines.append(f"Continue reasons: {'; '.join(continue_reasons)}")
    if failed_reasons:
        lines.append(f"Failed reasons: {'; '.join(failed_reasons)}")
    lines.append("")

    lines.append("## DECISION")
    lines.append("Choose exactly one:")
    lines.append(
        "1. complete_step — you now have enough: call add_artifact then complete_step."
    )
    lines.append(
        "2. continue_step(reason=...) — the objective is still completable: record "
        "the reason, then keep exploring and committing clues (commit_clue), and "
        "finally add_artifact + complete_step."
    )
    lines.append(
        "3. give up — the objective itself is uncompletable (e.g. a file is too "
        "large to read in full): record a failure conclusion with add_artifact, "
        "then terminate_subtree(reason=..., derive_axiom=<a confirmed negative "
        "fact>)."
    )
    if round_n >= max_recovery_rounds:
        lines.append("")
        lines.append(
            "This is the last recovery round — if you cannot produce an artifact "
            "now, you MUST give up (terminate_subtree)."
        )
    return "\n".join(lines)


def should_recover(step: StepSpec, result: StepResult) -> bool:
    """Return True when *step* ended unsuccessfully and is recovery-enabled."""
    return Feature.RECOVERY.value in step.features and not result.ok


def attach_recovery_hooks(
    steps: list[StepSpec],
    step_executor: Any,
    *,
    max_recovery_rounds: int = 3,
) -> list[StepSpec]:
    """Attach a ``RecoveryHook`` to each recovery-enabled step.

    Recovery enablement is declared per step type via ``StepType.features``
    and mirrored onto the ``StepSpec`` (phase 4); the runtime attaches the hook
    to any step carrying ``Feature.RECOVERY``.
    """
    for step in steps:
        if Feature.RECOVERY.value in step.features:
            hook = RecoveryHook(
                step_executor,
                max_recovery_rounds=max_recovery_rounds,
            )
            step.post_hooks.append(hook)
    return steps


def recovery_hook_factory(ctx: HookFactoryContext) -> RecoveryHook:
    """Registry factory: rebuild a ``RecoveryHook`` from a factory context.

    Pulls the pipeline's step executor (nullable) out of *ctx*.  The
    non-default ``max_recovery_rounds`` provenance is still open (phase-5 §7.3),
    so it keeps the phase-D default of 3.
    """
    return RecoveryHook(ctx.step_executor, max_recovery_rounds=3)
