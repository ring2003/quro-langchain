"""Thin StructuredTool wrappers that delegate to the controller layer.

Phase 6 deliverable (architecture §3.3).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool


def _parse_list(s: str) -> list[str]:
    """Parse comma-separated string to list."""
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _summarize_pipeline_result(result: Any) -> str:
    """Summarize a pipeline result for tool output."""
    if not hasattr(result, "step_results"):
        return str(result)[:2000]
    ok_count = sum(1 for r in result.step_results.values() if r.ok)
    total = len(result.step_results)
    lines = [f"Steps: {ok_count}/{total} OK"]
    for step_id, sr in result.step_results.items():
        status = "OK" if sr.ok else "FAIL"
        lines.append(f"  [{status}] {step_id}: {getattr(sr, 'summary', '')[:80]}")
    return "\n".join(lines)


def make_tool_surface(
    round_controller: Any,  # RoundController
    catalog_view: Any,  # IStepCatalogView
    handoff: Any,  # ICrossRoundHandoff
    gate: Any,  # PlanGate
) -> list[StructuredTool]:
    """Build the MetaPlanner tool set — thin wrappers, ZERO logic."""

    def create_step(
        step_id: str,
        objective: str,
        step_type: str = "",
        depends_on: str = "",
        skills: str = "",
        expected_output: str = "",
        validation: str = "",
    ) -> str:
        """Create a step specification directly (bypasses HTN solver)."""
        result = round_controller.build_step(
            step_id=step_id,
            objective=objective,
            step_type=step_type,
            depends_on=_parse_list(depends_on),
            skills=_parse_list(skills),
            expected_output=expected_output,
            validation=validation,
        )
        if not result.dependency_check.ok:
            return f"Error: {'; '.join(result.dependency_check.errors)}"
        return (
            f"Step '{step_id}' created (type={step_type or 'none'}, "
            f"features={result.resolved_features}, "
            f"warnings={result.warnings or 'none'})."
        )

    def list_operators() -> str:
        """List all HTN primitive operators currently registered."""
        summaries = catalog_view.describe_operators()
        return "\n".join(
            f"- {s.name}: {s.description[:60]} (skills={s.skill_pool}, access={s.access})"
            for s in summaries
        ) or "(no operators)"

    def get_artifact(artifact_id: str) -> str:
        """Retrieve an artifact by ID from current or prior rounds."""
        art = handoff.resolve_artifact(artifact_id)
        if art is None:
            return f"Artifact '{artifact_id}' not found in current or prior rounds."
        return str(art)[:2000]

    def submit_plan(plan_text: str) -> str:
        """Submit (or overwrite) this round's plan."""
        gate.plan_text = plan_text
        return f"Plan submitted ({len(plan_text)} chars)."

    def stop(conclusion: str = "") -> str:
        """Stop now (free stop) — no further plan is required."""
        gate.stop = True
        gate.conclusion = conclusion
        return "Stopped."

    def run_custom_pipeline(problem: str) -> str:
        """Execute the pipeline built from manually-created steps."""
        result = round_controller.execute()
        return _summarize_pipeline_result(result)

    return [
        StructuredTool.from_function(create_step, name="create_step"),
        StructuredTool.from_function(list_operators, name="list_operators"),
        StructuredTool.from_function(get_artifact, name="get_artifact"),
        StructuredTool.from_function(submit_plan, name="submit_plan"),
        StructuredTool.from_function(stop, name="stop"),
        StructuredTool.from_function(run_custom_pipeline, name="run_custom_pipeline"),
    ]
