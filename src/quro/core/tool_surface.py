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


def _build_meta_artifact_view(handoff: Any) -> Any:
    """Build a global ArtifactIndexView from the cross-round handoff.

    The MetaPlanner is the orchestrator above steps, so unlike a step-instance
    (whose view is scoped by ownership + ACL grants) it sees every artifact
    across all rounds and steps.  Uses ``build_global_artifact_index`` — the
    canonical ArtifactEntry/ArtifactIndexView projection shared with step
    agents (one semantic view, multiple prompt consumers — artifact_index §5).
    """
    from quro.context.artifact_index import build_global_artifact_index

    arts = getattr(handoff, "_artifacts", {})
    return build_global_artifact_index(list(arts.values()))


def _summarize_pipeline_result(
    result: Any,
    handoff_artifacts: dict[str, Any] | None = None,
) -> str:
    """Summarize a pipeline result for tool output.

    Includes a front-loaded ``## ARTIFACTS`` section (built via
    ``ArtifactIndexView``) so the MetaPlanner can ground artifact IDs
    without guessing.
    """
    if not hasattr(result, "step_results"):
        return str(result)[:2000]

    ok_count = sum(1 for r in result.step_results.values() if r.ok)
    total = len(result.step_results)

    # -- Collect artifacts via ArtifactIndexView ----------------------------
    # Merge current step results + handoff artifacts into one list, then
    # project through the canonical global artifact index.
    all_artifact_dicts: list[dict[str, Any]] = []
    for sr in result.step_results.values():
        for a in (getattr(sr, "artifacts", []) or []):
            if isinstance(a, dict) and a.get("artifact_id"):
                all_artifact_dicts.append(a)
    for aid, a in (handoff_artifacts or {}).items():
        if isinstance(a, dict) and a.get("artifact_id"):
            # Skip duplicates already in step results.
            if not any(x.get("artifact_id") == aid for x in all_artifact_dicts):
                all_artifact_dicts.append(a)

    view = None
    if all_artifact_dicts:
        from quro.context.artifact_index import build_global_artifact_index

        view = build_global_artifact_index(all_artifact_dicts)

    lines: list[str] = []

    # -- ARTIFACTS header (front-loaded for grounding) ----------------------
    if view and view.artifacts:
        lines.append("## ARTIFACTS")
        for entry in view.artifacts:
            lines.append(f"{entry.id}  (step {entry.step_id})")
        lines.append("")

    # -- Step Summary -------------------------------------------------------
    lines.append(f"Steps: {ok_count}/{total} OK")
    order = getattr(result, "order", None) or list(result.step_results.keys())
    for step_id in order:
        sr = result.step_results.get(step_id)
        if sr is None:
            lines.append(f"  [MISSING] {step_id}")
            continue
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
            view = _build_meta_artifact_view(handoff)
            known = (
                ", ".join(e.id for e in view.artifacts[:10])
                or "(none yet)"
            )
            return (
                f"Error: artifact not found: {artifact_id}. "
                f"Known artifacts: {known}"
            )
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
        return _summarize_pipeline_result(
            result,
            handoff_artifacts=getattr(handoff, "_artifacts", None),
        )

    def list_artifacts() -> str:
        """List all artifacts known to the cross-round handoff.

        Returns a compact table of artifact IDs, their originating step,
        kind, and summary so the MetaPlanner can resolve IDs without
        guessing.

        Uses the canonical ArtifactIndexView projection — the same
        artifact reality that step agents see (one semantic view).
        """
        view = _build_meta_artifact_view(handoff)
        if not view.artifacts:
            return "No artifacts recorded yet."

        rows: list[str] = ["id | step_id | kind | summary"]
        for entry in view.artifacts:
            rows.append(
                f"{entry.id} | {entry.step_id} | {entry.kind} | "
                f"{entry.summary[:80]}"
            )
        return "\n".join(rows)

    def list_manual_steps() -> str:
        """List all steps created via create_step."""
        pending = round_controller.scope.pending_steps
        if not pending:
            return "[]  (no manual steps — use create_step to add)"
        items = []
        for step_id, result in pending.items():
            spec = result.spec
            items.append({
                "step_id": step_id,
                "objective": spec.objective[:120] if spec else "",
                "step_type": spec.step_type if spec else "",
                "depends_on": list(spec.depends_on) if spec else [],
                "skills": list(spec.skills) if spec else [],
            })
        return json.dumps(items, indent=2)

    return [
        StructuredTool.from_function(create_step, name="create_step"),
        StructuredTool.from_function(list_operators, name="list_operators"),
        StructuredTool.from_function(list_manual_steps, name="list_manual_steps"),
        StructuredTool.from_function(get_artifact, name="get_artifact"),
        StructuredTool.from_function(list_artifacts, name="list_artifacts"),
        StructuredTool.from_function(submit_plan, name="submit_plan"),
        StructuredTool.from_function(stop, name="stop"),
        StructuredTool.from_function(run_custom_pipeline, name="run_custom_pipeline"),
    ]
