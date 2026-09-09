"""Backtracker — structural round-graph rebuild after abnormal termination.

Phase C of the architecture evolution (``docs/unsat-policy-loop/architecture/
evolution.md`` §2.7): a built-in, **domain-agnostic** runtime component (not a
registry ``Role``) that wakes on abnormal pipeline termination, rebuilds the
interrupted round's exploration graph **structurally**, and returns a
"context rebuild + next-step suggestion" to the **planner** (not the
MetaPlanner).

It consumes the session's public surface directly via the runtime-level
**round-object contract** (ADR-004 FR-8): steps (pre-step query +
``depends_on`` + ``access``) and artifacts (post-step).  ``checkpoint`` /
``backtrack`` are left untouched — they recover a consistent ``DomainState``
but drop the discarded branch's graph, so the backtracker reconstructs the
*interrupted* round from what the domain state still records.

This module is **pure and deterministic**: it reads a state dict and returns a
report.  Semantic re-planning stays with the planner; the (optional) internal
agent session only enriches structure, it does not re-plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quro.core.resources.graph import (
    DEPENDS_ON,
    DomainStateGraph,
    IResourceGraph,
)
from quro.core.resources.refs import ResourceRef
from quro.core.resources.resume import resume_domain_state


@dataclass
class RoundObject:
    """The runtime-level round-object contract (ADR-004 FR-8).

    Extracted from ``DomainState``: the steps (pre-step query, ``depends_on``,
    ``access``) and artifacts (post-step) of the interrupted round.
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    executing_step_id: str | None = None
    phase: str = ""
    recovery: dict[str, Any] = field(default_factory=dict)
    """Step-scoped recovery journal (blueprint §6): ``step_id -> {step_round,
    clues[], continue_reasons[], failed_reasons[]}``.  Carried so a lost
    artifact can be rebuilt from its clue chain."""

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "RoundObject":
        if not state:
            return cls()
        return cls(
            steps=[s for s in state.get("steps", []) if isinstance(s, dict)],
            artifacts=[a for a in state.get("artifacts", []) if isinstance(a, dict)],
            executing_step_id=state.get("executing_step_id"),
            phase=state.get("phase", ""),
            recovery=_copy_recovery(state.get("recovery")),
        )

    def step_ids(self) -> set[str]:
        return {str(s.get("step_id")) for s in self.steps if s.get("step_id")}

    def artifact_step_ids(self) -> set[str]:
        return {str(a.get("step_id")) for a in self.artifacts if a.get("step_id")}


@dataclass
class BacktrackReport:
    """Structural rebuild of an interrupted round + a next-step suggestion.

    Attributes:
        failed_step_id: The step whose abnormal termination woke the
            backtracker (the "last query").
        broken_deps: ``depends_on`` edges that reference an unknown step.
        missing_artifacts: Steps completed/in-progress without an artifact.
        blocked_steps: Pending steps whose dependencies are not yet complete.
        ready_steps: Pending steps whose dependencies are all complete
            (safe to resume immediately).
        next_step_suggestion: One-line guidance for the planner.
        context_rebuild: A structured text block the planner can read to
            rebuild context.
    """

    failed_step_id: str = ""
    broken_deps: list[dict[str, Any]] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    blocked_steps: list[str] = field(default_factory=list)
    ready_steps: list[str] = field(default_factory=list)
    next_step_suggestion: str = ""
    context_rebuild: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_step_id": self.failed_step_id,
            "broken_deps": list(self.broken_deps),
            "missing_artifacts": list(self.missing_artifacts),
            "blocked_steps": list(self.blocked_steps),
            "ready_steps": list(self.ready_steps),
            "next_step_suggestion": self.next_step_suggestion,
            "context_rebuild": self.context_rebuild,
        }


class Backtracker:
    """Rebuild the interrupted round's graph structurally (deterministic).

    Usage::

        report = Backtracker().rebuild(interrupted_state, failed_step_id="s3")
        planner.consume_backtrack(report)  # semantic re-planning stays here

    A ``graph`` may be injected (the shared edge table, architecture §4);
    when omitted it is derived from *state* on demand.
    """

    def __init__(
        self,
        graph: IResourceGraph | None = None,
        _restored_state: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._restored_state = _restored_state or {}

    @classmethod
    def from_snapshot(
        cls,
        store: Any,
        step_id: str,
        *,
        failed_step_id: str = "",
    ) -> "Backtracker":
        """Build a backtracker that reads the *restored* state (phase 2 §6).

        The graph is derived from the ``DomainState`` restored from the
        step-pre checkpoint snapshot — not from best-effort post-hoc
        reconstruction.
        """
        state = resume_domain_state(store, ResourceRef("step", step_id))
        graph = DomainStateGraph.from_state(state)
        return cls(graph=graph, _restored_state=state)

    def rebuild(
        self,
        state: dict[str, Any] | None,
        failed_step_id: str = "",
    ) -> BacktrackReport:
        """Analyze *state* (the interrupted round object) and return a report.

        When built via ``from_snapshot``, *state* may be omitted (``None``)
        and the restored state is used.
        """
        if state is None and self._restored_state:
            state = self._restored_state
        round_obj = RoundObject.from_state(state)
        graph = self._graph or DomainStateGraph.from_state(state)
        step_ids = round_obj.step_ids()
        artifact_step_ids = round_obj.artifact_step_ids()

        broken_deps = self._broken_deps(graph, step_ids)
        missing_artifacts = self._missing_artifacts(round_obj, artifact_step_ids)
        completed = self._completed(round_obj)
        blocked_steps, ready_steps = self._blocked_and_ready(graph, round_obj, completed)

        failed = failed_step_id or round_obj.executing_step_id or ""
        suggestion = self._suggest(
            failed=failed,
            broken_deps=broken_deps,
            missing_artifacts=missing_artifacts,
            blocked_steps=blocked_steps,
            ready_steps=ready_steps,
        )
        context = self._context_rebuild(
            round_obj=round_obj,
            failed=failed,
            broken_deps=broken_deps,
            missing_artifacts=missing_artifacts,
        )

        return BacktrackReport(
            failed_step_id=failed,
            broken_deps=broken_deps,
            missing_artifacts=missing_artifacts,
            blocked_steps=blocked_steps,
            ready_steps=ready_steps,
            next_step_suggestion=suggestion,
            context_rebuild=context,
        )

    # ------------------------------------------------------------------
    # Structural analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _broken_deps(graph: IResourceGraph, step_ids: set[str]) -> list[dict[str, Any]]:
        """``depends_on`` edges whose target step is unknown (from the edge table)."""
        broken: list[dict[str, Any]] = []
        for src, dst, rel in graph.edges():
            if rel != DEPENDS_ON:
                continue
            if dst.id not in step_ids:
                broken.append({
                    "step_id": src.id,
                    "missing_dep": dst.id,
                })
        return broken

    @staticmethod
    def _missing_artifacts(round_obj: RoundObject, artifact_step_ids: set[str]) -> list[str]:
        terminal = {"completed", "in_progress", "folded", "terminated"}
        return [
            str(s.get("step_id"))
            for s in round_obj.steps
            if s.get("step_id")
            and s.get("status") in terminal
            and s.get("step_id") not in artifact_step_ids
        ]

    @staticmethod
    def _completed(round_obj: RoundObject) -> set[str]:
        terminal = {"completed", "folded", "terminated"}
        return {
            str(s.get("step_id"))
            for s in round_obj.steps
            if s.get("status") in terminal
        }

    @staticmethod
    def _blocked_and_ready(
        graph: IResourceGraph, round_obj: RoundObject, completed: set[str]
    ) -> tuple[list[str], list[str]]:
        blocked: list[str] = []
        ready: list[str] = []
        for step in round_obj.steps:
            sid = str(step.get("step_id") or "")
            if not sid or step.get("status") != "pending":
                continue
            deps = graph.outgoing(ResourceRef("step", sid), DEPENDS_ON)
            if deps and any(d.id not in completed for d in deps):
                blocked.append(sid)
            else:
                ready.append(sid)
        return blocked, ready

    # ------------------------------------------------------------------
    # Suggestion + context rebuild
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest(
        *,
        failed: str,
        broken_deps: list[dict[str, Any]],
        missing_artifacts: list[str],
        blocked_steps: list[str],
        ready_steps: list[str],
    ) -> str:
        if broken_deps:
            missing = sorted({d["missing_dep"] for d in broken_deps})
            return (
                "fix depends_on: create or rename step(s) "
                + ", ".join(missing)
            )
        if missing_artifacts:
            return (
                "re-run step(s) "
                + ", ".join(missing_artifacts[:3])
                + " to produce a missing artifact before complete_step"
            )
        if blocked_steps:
            return "complete dependencies before resuming: " + ", ".join(blocked_steps[:3])
        if ready_steps:
            return "resume from ready step: " + ready_steps[0]
        if failed:
            return f"retry failed step '{failed}'"
        return "no structural break detected — verify goal facts"

    @staticmethod
    def _context_rebuild(
        *,
        round_obj: RoundObject,
        failed: str,
        broken_deps: list[dict[str, Any]],
        missing_artifacts: list[str],
    ) -> str:
        lines: list[str] = ["## Round Rebuild"]
        lines.append(f"phase: {round_obj.phase or '(unknown)'}")
        lines.append(f"last query: {failed or '(none)'}")
        lines.append(f"steps: {len(round_obj.steps)}, artifacts: {len(round_obj.artifacts)}")

        for step in round_obj.steps:
            sid = step.get("step_id")
            status = step.get("status", "?")
            deps = ", ".join(step.get("depends_on", [])) or "—"
            lines.append(f"- {sid} [{status}] depends_on=[{deps}]")

        clue_lines = _render_clues(round_obj.recovery)
        if clue_lines:
            lines.append("recovery clues (rebuild lost artifacts from these):")
            lines.extend(clue_lines)

        if broken_deps:
            lines.append("broken deps:")
            for b in broken_deps:
                lines.append(f"- {b['step_id']} -> missing {b['missing_dep']}")
        if missing_artifacts:
            lines.append("missing artifacts: " + ", ".join(missing_artifacts))
        return "\n".join(lines)


def _copy_recovery(recovery: Any) -> dict[str, Any]:
    """Shallow-copy the recovery journal map (list values copied)."""
    out: dict[str, Any] = {}
    for step_id, journal in (recovery or {}).items():
        if not isinstance(journal, dict):
            continue
        copied: dict[str, Any] = {}
        for key, value in journal.items():
            copied[key] = list(value) if isinstance(value, list) else value
        out[str(step_id)] = copied
    return out


def _render_clues(recovery: dict[str, Any]) -> list[str]:
    """Render the committed clues per step for the context rebuild."""
    lines: list[str] = []
    for step_id, journal in (recovery or {}).items():
        for clue in (journal or {}).get("clues", []):
            if isinstance(clue, dict) and clue.get("text"):
                lines.append(f"- [{step_id}] {clue['text']}")
    return lines
