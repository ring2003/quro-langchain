"""Session — the framework-side single-round primitive (D2).

After g/f separation, ``PipelineRunner`` is the executor and the agent-driven
scratchpad loop is the planner.  ``Session`` is the framework's assembled,
single-round primitive that the agent loop drives (``meta-plan-prompt-and-ppf.md``
§3):

::

    pef = build_pipeline(plan + PPF)          # g: plan -> [StepSpec]
    artifact = Session.run(pef)               # f: execute
    goal_status = Session.evaluate(pef)       # framework fact layer
    rounds.append(RoundRecord(actual=...))    # framework writes (read-only)

It is **not** the meta-planner: it keeps no cross-round plan state and reads no
verdict.  ``run`` is pure ``f`` (``PipelineRunner``); ``evaluate`` is the fact
layer (``goal_status_from_results``); ``report`` assembles the user-facing
report artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quro.planner.goal_status import FactExtractor, UNSATDiagnosis, goal_status_from_results
from quro.pipeline.core import Pipeline, PipelineConfig, PipelineResult, PipelineRunner


@dataclass
class SessionResult:
    """The product of one framework round: fact layer + report, no verdict."""

    pipeline: PipelineResult
    goal_status: UNSATDiagnosis


class Session:
    """Framework-side single-round primitive.

    Args:
        step_executor: The ``StepExecutor`` (e.g. ``StepAdapter``).
        pipeline_config: Per-run ``PipelineConfig``.
        fact_extractor: Fact extraction strategy for ``evaluate``.
    """

    def __init__(
        self,
        step_executor: Any,
        *,
        pipeline_config: PipelineConfig | None = None,
        fact_extractor: FactExtractor | None = None,
    ) -> None:
        self._step_executor = step_executor
        self._pipeline_config = pipeline_config or PipelineConfig()
        self._fact_extractor = fact_extractor or FactExtractor()

    # ------------------------------------------------------------------- run
    def run(
        self,
        pipeline: Pipeline,
        problem: str,
        *,
        context_results: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline — pure ``f`` (closure preserved).

        The output is a ``PipelineResult`` (whose step results carry the
        artifacts); no ``[StepSpec]`` is emitted here.
        """
        runner = PipelineRunner(
            pipeline,
            step_executor=self._step_executor,
            config=self._pipeline_config,
        )
        return runner.run(problem, context_results=context_results)

    # ------------------------------------------------- IPipelineController.execute
    def execute(self, batch: Any) -> PipelineResult:
        """Execute a batch of steps — ``IPipelineController`` implementation.

        ``batch`` is a ``RoundScope`` (or any object with ``pending_steps``
        ``dict[str, StepBuildResult]``).  Constructs a ``Pipeline`` from the
        pending steps and runs it.
        """
        from quro.core.conformance import ConformanceChecker

        # Conformance check: validate round batch before execution.
        checker = ConformanceChecker(testing=True)
        violations = checker.check_round_batch(batch)
        if violations:
            import logging
            logging.getLogger(__name__).warning(
                "Conformance round-batch violations: %s",
                "; ".join(v.message for v in violations),
            )

        pending = batch.pending_steps
        steps = [result.spec for result in pending.values()]
        if not steps:
            return PipelineResult(
                ok=True,
                step_results={},
                order=[],
            )
        pipeline = Pipeline(
            name="round_pipeline",
            steps=steps,
            config=self._pipeline_config,
        )
        return self.run(pipeline, "")

    # --------------------------------------------------------------- evaluate
    def evaluate(
        self,
        result: PipelineResult,
        goal_facts: list[str],
    ) -> UNSATDiagnosis:
        """Compute the fact layer ``goal_status`` (objective, reproducible)."""
        return goal_status_from_results(
            result,
            goal_facts=goal_facts,
            extractor=self._fact_extractor,
            pef_empty=False,
            round_index=0,
        )

    # ---------------------------------------------------------------- report
    @staticmethod
    def report(
        result: PipelineResult,
        goal_status: UNSATDiagnosis,
    ) -> str:
        """Assemble the user-facing report text (the loop's exit artifact).

        ``goal_status`` sat/unsat facts are rendered as facts, not as a
        control signal — the runtime reads none of it to force a stop.
        """
        ok_count = sum(1 for r in result.step_results.values() if r.ok)
        total = len(result.step_results)
        lines: list[str] = [
            f"Steps: {ok_count}/{total} OK",
            f"SAT facts: {sorted(goal_status.sat_facts)}",
            f"UNSAT facts: {sorted(goal_status.unsat_facts)}",
        ]
        if goal_status.cause:
            lines.append(f"Cause: {goal_status.cause}")
        return "\n".join(lines)
