"""CrossRoundHandoff — ICrossRoundHandoff implementation.

Stores artifacts and step references from prior rounds.
Qualified id form: 'r<round_idx>:<step_id>'.

Phase 4 deliverable (architecture §3.2).
"""

from __future__ import annotations

from typing import Any

from quro.core.protocols import ICrossRoundHandoff, StepBuildResult


class CrossRoundHandoff:
    """ICrossRoundHandoff implementation.

    Stores artifacts and step references from prior rounds.
    Qualified id form: 'r<round_idx>:<step_id>'.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, Any] = {}
        self._step_refs: dict[str, StepBuildResult] = {}

    def carry_artifact(self, round_idx: int, artifact_id: str, artifact: Any) -> None:
        self._artifacts[artifact_id] = artifact

    def resolve_artifact(self, artifact_id: str) -> Any | None:
        return self._artifacts.get(artifact_id)

    def carry_step_ref(self, round_idx: int, step_id: str, result: StepBuildResult) -> None:
        qualified = f"r{round_idx}:{step_id}"
        self._step_refs[qualified] = result

    def resolve_step_ref(self, qualified_id: str) -> StepBuildResult | None:
        return self._step_refs.get(qualified_id)


def carry_result_artifacts(
    handoff: ICrossRoundHandoff,
    round_idx: int,
    result: Any,
) -> None:
    """Carry every artifact produced by an executed pipeline into *handoff*.

    Walks ``result.step_results`` (a ``PipelineResult``) and carries each
    step's artifact dicts by their ``artifact_id``, so outer consumers (the
    round / meta-planner ``get_artifact`` tools) can resolve them.
    """
    step_results = getattr(result, "step_results", None) or {}
    if not isinstance(step_results, dict):
        return
    for step_result in step_results.values():
        for art in list(getattr(step_result, "artifacts", None) or []):
            if isinstance(art, dict) and art.get("artifact_id"):
                handoff.carry_artifact(round_idx, str(art["artifact_id"]), art)
