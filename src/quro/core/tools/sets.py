"""Resource-scoped toolsets — semantic tools organised by resource domain.

Each toolset subclasses :class:`~quro.core.tools.spec.Toolset`, declares its
``resource_kind``, and implements tools as public methods.  The method
signature is the tool schema, the docstring is the description, and the method
body is the handler (routed to the kernel via ``session.call``).  Read / write
handlers run the Layer 2 ACL check at entry (architecture §5).

This replaces the old hand-written ``make_all_tools`` / ``make_research_tools``
/ ``make_recovery_tools`` tables in ``quro.core.session.tools`` with a single
declarative surface — a tool's name, schema and handler now live in one place.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from quro.core.resources import (
    Decision,
    ResourceRef,
    acl_for_state,
    principal_from_state,
)
from quro.core.tools.spec import (
    RESOURCE_KIND_ARTIFACT,
    RESOURCE_KIND_CLUE,
    RESOURCE_KIND_PLAN,
    RESOURCE_KIND_STEP,
    Toolset,
    format_response,
)


def _fmt_resp(name: str, resp: Any) -> str:
    return format_response(name, resp)


# ---------------------------------------------------------------------------
# StepTool — the step resource (DomainState.steps entries)
# ---------------------------------------------------------------------------


class StepTool(Toolset):
    """Tools over the step resource: create / read / update / complete."""

    resource_kind = RESOURCE_KIND_STEP

    def create_step(
        self,
        session: Any,
        step_id: str,
        objective: str,
        step_type: str = "",
        skills: str = "",
    ) -> str:
        """Create a new step specification with a unique step ID, objective,
        and the step type it instantiates. ``step_type`` must name a known
        StepType (the step type IS the execution unit). ``skills`` is a
        comma-separated subset of the step type's
        skill pool. The same step type may be reused across several steps."""
        args: dict[str, Any] = {"step_id": step_id, "objective": objective}
        if step_type:
            args["step_type"] = step_type
        if skills:
            args["skills"] = [x.strip() for x in skills.split(",") if x.strip()]
        res = session.call("create_step", args)
        if not res.ok:
            return _fmt_resp("create_step", res)
        return f"Step {step_id} created: {objective}"

    def get_step(self, session: Any, step_id: str) -> str:
        """Retrieve details of a prior step by its step_id. Returns step_id,
        objective, status, depends_on, access, inputs, expected_output,
        validation, checklist summary, and artifact_ids produced by this step."""
        denied = _acl_deny(session, ResourceRef("step", step_id), "read")
        if denied:
            return denied
        state = session.state or {}
        for s in state.get("steps", []):
            if s.get("step_id") == step_id:
                _record_access(session, "step", step_id)
                step_arts = [
                    a["artifact_id"] for a in state.get("artifacts", [])
                    if a.get("step_id") == step_id
                ]
                result = {
                    "step_id": s.get("step_id"),
                    "objective": s.get("objective", ""),
                    "status": s.get("status", ""),
                    "depends_on": s.get("depends_on", []),
                    "access": s.get("access"),
                    "inputs": s.get("inputs", []),
                    "expected_output": s.get("expected_output", []),
                    "validation": s.get("validation", []),
                    "checklist": [
                        {"item_id": c.get("item_id", ""), "description": c.get("description", ""), "done": c.get("done", False)}
                        for c in s.get("checklist", [])
                    ],
                    "artifact_ids": step_arts,
                }
                return json.dumps(result, ensure_ascii=False, indent=2)
        return f"Error: step not found: {step_id}"

    def update_step(self, session: Any, step_id: str, field: str, value: str) -> str:
        """Update one field of a step. ``field`` is one of: inputs,
        expected_output, validation, dependencies, checklist (appended),
        depends_on (comma-separated, replaces), access (must be 'hints'),
        hint_max_chars (integer 80-400)."""
        res = session.call("update_step", {"step_id": step_id, "field": field, "value": value})
        if not res.ok:
            return _fmt_resp("update_step", res)
        return f"Updated {step_id}.{field}"

    def complete_step(self, session: Any, step_id: str = "") -> str:
        """Mark a step as completed. PREREQUISITE: add_artifact MUST be called
        first (the gate check rejects steps with no artifacts). If step_id is
        empty, the current executing step is used automatically."""
        if not step_id:
            state = session.state or {}
            step_id = state.get("executing_step_id", step_id)
        denied = _acl_deny(session, ResourceRef("step", step_id), "edit")
        if denied:
            return denied
        access_log = getattr(session, "_round_access_log", None)
        call_args: dict[str, Any] = {"step_id": step_id}
        if access_log is not None:
            call_args["access_log"] = list(access_log)
        res = session.call("complete_step", call_args)
        if not res.ok:
            error_lines = [f"Error: {res.error}"]
            if res.explanation:
                error_lines.append(res.explanation)
            if "DEP_ACCESS_REQUIRED" in (res.error or ""):
                error_lines.append(
                    "\nRECOVERY: You MUST complete these steps BEFORE calling complete_step again:\n"
                    "  1. Call get_artifact(artifact_id) OR get_step(step_id) for each missing dependency\n"
                    "  2. Call add_artifact again with the fetched artifact_ids cited in evidences or body\n"
                    "  3. Then retry complete_step\n"
                    "\nDo NOT call complete_step again without first fetching the missing dependencies."
                )
            return "\n".join(error_lines)
        state = session.state or {}
        step_arts = [
            a.get("artifact_id")
            for a in state.get("artifacts", [])
            if a.get("step_id") == step_id and a.get("artifact_id")
        ]
        artifacts_line = f" Artifacts: {', '.join(step_arts)}" if step_arts else ""
        return f"Step {step_id} marked completed.{artifacts_line}"

    def finalize_step(self, session: Any) -> str:
        """Finalize the current step specification and move to execution phase."""
        res = session.call("finalize_step", {})
        if not res.ok:
            return _fmt_resp("finalize_step", res)
        return "Step finalized. Moving to execution."

    def terminate_subtree(self, session: Any, reason: str, derive_axiom: str = "") -> str:
        """Declare the current subtree as unsolvable/exhausted and fold it up.
        ``derive_axiom`` optionally preserves a confirmed negative fact across
        the fold boundary."""
        from quro.core.fold import handle_terminate_subtree

        result = handle_terminate_subtree(session, reason, derive_axiom)
        if result.get("status") == "error":
            return f"Error: {result.get('reason', 'termination failed')}"
        return f"Subtree terminated: {reason[:200]}"


# ---------------------------------------------------------------------------
# ArtifactTool — the artifact resource
# ---------------------------------------------------------------------------


class ArtifactTool(Toolset):
    """Tools over the artifact resource: add / read."""

    resource_kind = RESOURCE_KIND_ARTIFACT

    def __init__(self, resource_store: Any = None) -> None:
        self._resource_store = resource_store

    def add_artifact(self, session: Any, summary: str, kind: str = "analysis", body: str = "", evidences: str = "") -> str:
        """Attach evidence of work done to the current step. REQUIRED before
        complete_step. kind: 'file' (body=path to an existing file, PREFERRED),
        'analysis', 'intermediate', 'test_result'. evidences: references or
        paths, one per line."""
        state = session.state or {}
        denied = _acl_deny(
            session,
            ResourceRef("step", state.get("executing_step_id") or ""),
            "write",
        )
        if denied:
            return denied
        kind = (kind or "analysis").strip().lower()
        if kind == "file":
            if not body or not body.strip():
                return "Error: kind='file' requires body to be the path of an existing file"
            path = Path(body).expanduser()
            if not path.is_file():
                return f"Error: file artifact body must be a path to an existing file; path not found: {body}"
            body = str(path)
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        art: dict[str, Any] = {
            "artifact_id": artifact_id,
            "summary": summary,
            "kind": kind,
            "body": body,
            "evidences": [e.strip() for e in evidences.split("\n") if e.strip()] if evidences else [],
        }
        if self._resource_store is not None:
            self._resource_store.save(ResourceRef("artifact", artifact_id), art)
        args: dict[str, Any] = {"artifact_id": artifact_id, "summary": summary, "kind": kind}
        if body:
            args["body"] = body
        if evidences:
            args["evidences"] = art["evidences"]
        session.call("add_artifact", args)
        return f"Artifact recorded: {artifact_id} — {summary}"

    def get_artifact(self, session: Any, artifact_id: str) -> str:
        """Retrieve full content of a prior artifact by its artifact_id.
        Returns the complete artifact record including body; for kind='file'
        the file content is read directly and returned as 'content'."""
        denied = _acl_deny(session, ResourceRef("artifact", artifact_id), "read")
        if denied:
            return denied
        if self._resource_store is not None:
            art = self._resource_store.load(ResourceRef("artifact", artifact_id))
            if art:
                _record_access(session, "artifact", artifact_id)
                return json.dumps(_load_artifact(art), ensure_ascii=False, indent=2)
        state = session.state or {}
        for a in state.get("artifacts", []):
            if a.get("artifact_id") == artifact_id:
                _record_access(session, "artifact", artifact_id)
                return json.dumps(_load_artifact(a), ensure_ascii=False, indent=2)
        return f"Error: artifact not found: {artifact_id}"


# ---------------------------------------------------------------------------
# PlanningTool — planner state fields (interpretation / plan)
# ---------------------------------------------------------------------------


class PlanningTool(Toolset):
    """Tools over the planner's session state (interpretation / plan)."""

    resource_kind = RESOURCE_KIND_PLAN

    def set_interpretation(self, session: Any, text: str) -> str:
        """Record your interpretation of the problem."""
        session.call("set_interpretation", {"text": text})
        return "Interpretation recorded."

    def set_plan(self, session: Any, plan: str) -> str:
        """Record the high-level plan."""
        session.call("set_plan", {"plan": plan})
        return "Plan recorded."

    def confirm_understanding(self, session: Any) -> str:
        """Confirm understanding and proceed to step specification. Requires
        interpretation and plan to be set."""
        res = session.call("confirm_understanding", {})
        if not res.ok:
            return _fmt_resp("confirm_understanding", res)
        return "Understanding confirmed. Moving to step specification."


# ---------------------------------------------------------------------------
# EvaluationTool — evaluation verdict
# ---------------------------------------------------------------------------


class EvaluationTool(Toolset):
    """Tools over the evaluation phase."""

    def submit_evaluation(self, session: Any, proposal: str, rationale: str) -> str:
        """Submit evaluation result. proposal: complete, continue, or needs_work."""
        valid = {"complete", "continue", "needs_work"}
        proposal = proposal.strip().lower()
        if proposal not in valid:
            return f"Error: proposal must be one of {valid}"
        session.call("submit_evaluation", {"proposal": proposal, "rationale": rationale})
        return f"Evaluation submitted: {proposal}"


# ---------------------------------------------------------------------------
# ClarificationTool — planner/worker clarification channel
# ---------------------------------------------------------------------------


class ClarificationTool(Toolset):
    """Tools over the clarification channel."""

    def request_clarification(self, session: Any, message: str) -> str:
        """Request additional information or clarification from the planner."""
        if not message.strip():
            return "Error: message cannot be empty"
        session.call("request_clarification", {"message": message})
        return f"Clarification requested from planner: {message[:200]}"

    def resolve_clarification(self, session: Any, index: int) -> str:
        """Mark a pending clarification request as resolved."""
        session.call("resolve_clarification", {"index": index})
        return f"Clarification {index} marked resolved."


# ---------------------------------------------------------------------------
# SessionTool — checkpoint / backtrack (kernel-intercepted)
# ---------------------------------------------------------------------------


class SessionTool(Toolset):
    """Tools over the session lifecycle (checkpoint / backtrack)."""

    def checkpoint(self, session: Any, label: str) -> str:
        """Save a named checkpoint for backtracking."""
        res = session.call("checkpoint", {"label": label})
        if not res.ok:
            return f"Checkpoint failed: {res.error}"
        return f"Checkpoint '{label}' set."

    def backtrack(self, session: Any, to_label: str, reason: str) -> str:
        """Backtrack to a named checkpoint, discarding events after it."""
        res = session.call("backtrack", {"to_label": to_label, "reason": reason})
        if not res.ok:
            return f"Backtrack failed: {res.error}"
        return f"Backtracked to '{to_label}': {reason}"


# ---------------------------------------------------------------------------
# ReportTool — final report assembly
# ---------------------------------------------------------------------------


class ReportTool(Toolset):
    """Research-domain tool: assemble the report."""

    def assemble_report(self, session: Any, path: str, title: str = "") -> str:
        """Record the final report file path (merging per-phase chunks)."""
        res = session.call("assemble_report", {"path": path, "title": title})
        if not res.ok:
            return _fmt_resp("assemble_report", res)
        return f"Report recorded: {path}"


# ---------------------------------------------------------------------------
# RecoveryTool — interruption recovery journal
# ---------------------------------------------------------------------------


class RecoveryTool(Toolset):
    """Research-domain tools: interruption recovery (clue chain)."""

    resource_kind = RESOURCE_KIND_CLUE

    def commit_clue(self, session: Any, clue: str, step_id: str = "") -> str:
        """Persist one compact finding (a clue) for the current step. Clues are
        append-only and step-scoped; commit one small finding per call, so an
        interruption cannot lose it. step_id defaults to the current step."""
        state = session.state or {}
        target_step = step_id or state.get("executing_step_id") or ""
        denied = _acl_deny(session, ResourceRef("step", target_step), "write")
        if denied:
            return denied
        res = session.call("commit_clue", {"clue": clue, "step_id": step_id})
        if not res.ok:
            return _fmt_resp("commit_clue", res)
        return "Clue committed."

    def continue_step(self, session: Any, reason: str) -> str:
        """Declare that you need more exploration to complete this step.
        Records the reason and resumes from the committed clues."""
        res = session.call("continue_step", {"reason": reason})
        if not res.ok:
            return _fmt_resp("continue_step", res)
        return "Continue recorded. Resume exploration from the committed clues."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _acl_deny(session: Any, ref: ResourceRef, action: str) -> str | None:
    """Run the Layer 2 ACL check at call time; return an error on deny.

    The check happens **before** the handler reaches ``session.call``: a
    denied read/write never touches the kernel (architecture §5/§7).
    """
    state = session.state or {}
    principal = principal_from_state(state)
    verdict = acl_for_state(state).check(principal, ref, action)
    if verdict.decision is Decision.DENY:
        return f"Error: access denied ({verdict.reason})"
    return None


def _record_access(session: Any, kind: str, entity_id: str) -> None:
    """Record an access for audit only (no longer an enforcement input)."""
    access_log = getattr(session, "_round_access_log", None)
    if access_log is not None:
        access_log.add((kind, entity_id))


def _load_artifact(art: dict[str, Any]) -> dict[str, Any]:
    """Resolve a stored artifact for retrieval (kind='file' → read content)."""
    result = dict(art)
    if art.get("kind") == "file":
        path = art.get("body", "")
        try:
            result["content"] = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            result["content"] = f"<unable to read file at {path}: {e}>"
    return result


# ---------------------------------------------------------------------------
# Toolset assembly — framework built-ins vs domain extensions
# ---------------------------------------------------------------------------


def framework_toolsets(resource_store: Any = None) -> list[Toolset]:
    """Framework built-in toolsets — every framework semantic tool.

    These are the generic semantic tools the framework owns: step CRUD,
    artifact CRUD, planner state, evaluation, clarification, session
    lifecycle, and the interruption-recovery journal.  They are framework
    content, not domain content — whether a given step receives each tool is
    the ``ToolCoordinator``'s allocation decision (contract / lifecycle /
    ``StepType.tools``).
    """
    return [
        StepTool(),
        ArtifactTool(resource_store=resource_store),
        PlanningTool(),
        EvaluationTool(),
        ClarificationTool(),
        SessionTool(),
        RecoveryTool(),
    ]


def domain_toolsets(domain: Any) -> list[Toolset]:
    """Domain-extension toolsets, derived from the domain's declarations.

    A research domain (one declaring a phase vocabulary via
    ``phase_for_step_type``) contributes the report-assembly tool.  Recovery
    tools are framework content (see :func:`framework_toolsets`), allocated by
    ``Feature.RECOVERY`` — not by a ``recovery_step_types`` declaration.
    """
    toolsets: list[Toolset] = []
    if hasattr(domain, "phase_for_step_type"):
        toolsets.append(ReportTool())
    return toolsets
