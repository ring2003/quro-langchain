"""Prompt blocks — block model, cognitive ordering, and block renderers.

A step prompt is two containers (system / user), each an ordered list of
sub-blocks.  A block carries ``name`` / ``kind`` / ``content`` / ``role``;
``kind`` drives the cognitive order and trim priority.  Renderers are
registered with ``@user_block`` / ``@system_block``; a domain declares which
blocks compose each phase with ``@phase_blocks``.  The ``PromptCoordinator``
(``coordinator.py``) collects blocks from hooks, orders them, and assembles the
containers.

This module replaces ``projector.project_state``'s per-phase hard-coding: the
phase-specific user view is now *declared* by the domain, rendered uniformly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from quro.context.artifact_index import ArtifactIndexView, build_artifact_index
from quro.core.resources.acl import acl_for_state, principal_from_state

BlockRenderer = Callable[..., str]

_BLOCKS: dict[str, BlockRenderer] = {}
_BLOCK_KINDS: dict[str, str] = {}

HINT_MAX_CHARS = int(os.environ.get("QURO_HINT_MAX_CHARS", "160"))
HINTS_MAX_ITEMS = int(os.environ.get("QURO_HINTS_MAX_ITEMS", "8"))
HINTS_BUDGET_CHARS = max(0, int(os.environ.get("QURO_HINTS_BUDGET_CHARS", "0")))


@dataclass
class Block:
    """A named prompt fragment with a cognitive kind and container role."""

    name: str
    kind: str
    content: str
    role: str = "user"  # "system" | "user"


# Cognitive order: lower index = earlier in the prompt (design §3).
KIND_ORDER: dict[str, int] = {
    # system
    "identity": 0,
    "duties": 1,
    "constraints": 2,
    "output_format": 3,
    "project_scope": 4,
    # user — context first, then resources, then the focused task last
    "problem": 10,
    "plan": 11,
    "interpretation": 12,
    "evidence": 13,
    "report_chunks": 14,
    "module_inventory": 15,
    "completed_steps": 16,
    "steering_artifacts": 17,  # steering act's full-content artifact view (evidence band)
    "hints": 20,
    "own_history": 21,  # NEW: own prior-round artifacts
    "tools": 22,
    "skills": 23,
    "tips": 24,
    "steering_context": 29,  # steering act's dynamic prompt (objective band)
    "objective": 30,
    "next_instruction": 31,
    "custom": 40,
}


def _register(name: str, kind: str, fn: BlockRenderer) -> BlockRenderer:
    _BLOCKS[name] = fn
    _BLOCK_KINDS[name] = kind or name
    return fn


def user_block(name: str, kind: str = "") -> Callable[[BlockRenderer], BlockRenderer]:
    """Register a user-container block renderer (decorator)."""
    def decorator(fn: BlockRenderer) -> BlockRenderer:
        return _register(name, kind or name, fn)
    return decorator


def system_block(name: str, kind: str = "") -> Callable[[BlockRenderer], BlockRenderer]:
    """Register a system-container block renderer (decorator)."""
    def decorator(fn: BlockRenderer) -> BlockRenderer:
        return _register(name, kind or name, fn)
    return decorator


def phase_blocks(mapping: dict[str, list[str]]) -> Callable[[type], type]:
    """Class decorator declaring a domain's ``phase → block list`` mapping."""
    def decorator(cls: type) -> type:
        cls.PHASE_BLOCKS = dict(mapping)
        return cls
    return decorator


def block_kind(name: str) -> str:
    registered = _BLOCK_KINDS.get(name)
    if registered:
        return registered
    # Hook-provided blocks (objective / skills / project_scope / next_instruction)
    # have no renderer; their kind is the KIND_ORDER entry itself, not "custom".
    return name if name in KIND_ORDER else "custom"


def registered_blocks() -> list[str]:
    """Return the names of all registered (state-derived) block renderers."""
    return list(_BLOCKS)


def render_block(name: str, state: dict[str, Any], *, phase: str, budget: int, principal: str | None) -> str:
    fn = _BLOCKS.get(name)
    if fn is None:
        raise RuntimeError(f"Unknown block {name!r} — registered blocks: {sorted(_BLOCKS)}")
    return fn(state, phase=phase, budget=budget, principal=principal)


def render_phase(
    state: dict[str, Any],
    phase: str,
    blocks: list[str],
    *,
    budget: int,
    principal: str | None,
) -> str:
    """Render *blocks* in declaration order, skipping empties."""
    parts: list[str] = []
    for name in blocks:
        content = render_block(name, state, phase=phase, budget=budget, principal=principal)
        if content:
            parts.append(content)
    result = "\n".join(parts)
    if len(result) > budget:
        result = result[:budget] + "\n\n---\n**CONTEXT TRUNCATED**"
    return result


# ---------------------------------------------------------------------------
# State helpers (moved from projector.py)
# ---------------------------------------------------------------------------


def current_step(state: dict[str, Any]) -> dict[str, Any] | None:
    current_id = state.get("executing_step_id") or state.get("designing_step_id")
    if not current_id:
        return None
    for s in state.get("steps", []):
        if s["step_id"] == current_id:
            return s
    return None


def _filter_problem_for_principal(problem_text: str, principal: str | None) -> str:
    """Drop CONSTRAINT lines that target a different step type."""
    if not principal:
        return problem_text
    known_roles = {"plan", "implement", "specify", "evaluate", "explore"}
    lines = problem_text.split("\n")
    filtered: list[str] = []
    in_constraints = False
    for line in lines:
        stripped = line.strip()
        if "**CONSTRAINTS**" in stripped or stripped.upper().startswith("CONSTRAINTS"):
            in_constraints = True
            filtered.append(line)
            continue
        if in_constraints:
            if stripped.startswith(("##", "**")) and not stripped.startswith("- "):
                in_constraints = False
                filtered.append(line)
                continue
            if stripped.startswith(("- ", "* ", "+ ")):
                mentions_other = any(r in line for r in known_roles if r != principal)
                if mentions_other and principal not in line:
                    continue
                filtered.append(line)
                continue
            filtered.append(line)
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _render_hints_table(rows: list[dict[str, Any]], dep_ids: list[str]) -> str:
    lines = [
        "\n## HINTS (stateful deps — pull full text via tools)",
        "Access mode: hints. Full bodies are NOT injected.",
        "Use get_artifact(artifact_id) or get_step(step_id) to read more.",
        "",
        "| artifact_id | step_id | kind | chars | preview |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['artifact_id']} | {r['step_id']} | {r['kind']} | {r['chars_total']} | {r['preview']} |"
        )
    lines.append(f"\ndepends_on: {', '.join(dep_ids)}")
    return "\n".join(lines)


def _build_hints(state: dict[str, Any], step: dict[str, Any], budget: int) -> str:
    hint_max = step.get("hint_max_chars") or HINT_MAX_CHARS
    hints_budget = HINTS_BUDGET_CHARS or min(2000, max(400, budget // 5))
    max_items = HINTS_MAX_ITEMS
    dep_ids = step.get("depends_on", [])

    # Dependency artifacts come from the unified ArtifactIndexView (the ACL
    # grant check + dedup are already done there), not by hand.
    view = _artifact_view(state)
    dep_artifacts = [e for e in view.index.artifacts if e.step_id != step.get("step_id")]

    dep_rows = dep_artifacts[:max_items]
    if not dep_rows:
        return ""

    rows: list[dict[str, Any]] = []
    for e in dep_rows:
        source = e.body or e.summary
        preview = source[:hint_max].replace("\n", " ").rstrip()
        if len(source) > hint_max:
            preview += "\u2026"
        rows.append({
            "artifact_id": e.id,
            "step_id": e.step_id,
            "kind": e.kind,
            "chars_total": len(source),
            "preview": preview,
        })

    while True:
        serialized = _render_hints_table(rows, dep_ids)
        if len(serialized) <= hints_budget:
            break
        if len(rows) > 1:
            rows.pop()
        elif rows:
            rows[0]["preview"] = rows[0]["preview"][:max(0, len(rows[0]["preview"]) // 2)]
            if not rows[0]["preview"]:
                rows.pop()
        else:
            break

    if not rows:
        return ""
    return _render_hints_table(rows, dep_ids)


# ---------------------------------------------------------------------------
# Built-in user blocks
# ---------------------------------------------------------------------------


@user_block("problem")
def _problem(state, *, phase, budget, principal):
    problem = str(state.get("problem", "") or "").strip()
    if principal:
        problem = _filter_problem_for_principal(problem, principal)
    return f"## Problem\n{problem[:2000]}" if problem else ""


@user_block("plan")
def _plan(state, *, phase, budget, principal):
    plan = state.get("plan") or state.get("exploration_plan") or ""
    plan = str(plan).strip()
    return f"## Plan\n{plan[:2000]}" if plan else ""


@user_block("interpretation")
def _interpretation(state, *, phase, budget, principal):
    interp = str(state.get("interpretation", "") or "").strip()
    return f"## Interpretation\n{interp[:500]}" if interp else ""


@user_block("current_step")
def _current_step_block(state, *, phase, budget, principal):
    step = current_step(state)
    if not step:
        return ""
    lines = [
        f"## Current Step: {step['step_id']}",
        f"Objective: {step.get('objective', '')}",
        f"\nWhen finished, call complete_step(step_id='{step['step_id']}').",
    ]
    if step.get("depends_on"):
        lines.append(f"depends_on: {step['depends_on']}")
    if step.get("expected_output"):
        eo = step["expected_output"]
        if isinstance(eo, (list, tuple)):
            lines.append("Expected Output:\n" + "\n".join(f"- {o}" for o in eo))
        else:
            lines.append(f"Expected Output: {eo}")
    return "\n".join(lines)


@user_block("completed_steps")
def _completed_steps(state, *, phase, budget, principal):
    done = [s for s in state.get("steps", []) if s.get("status") == "completed"]
    if not done:
        return ""
    lines = [f"## Completed Steps: {len(done)}"]
    for s in done:
        lines.append(f"- {s['step_id']}: {s.get('objective', '')[:100]}")
    return "\n".join(lines)


@user_block("hints")
def _hints(state, *, phase, budget, principal):
    step = current_step(state)
    if not step:
        return ""
    if not (step.get("depends_on") and step.get("access") == "hints"):
        return ""
    return _build_hints(state, step, budget)


@user_block("own_history")
def _own_history(state, *, phase, budget, principal):
    """Show this step's own prior-round artifacts to the step agent.

    Closes the visibility gap: the step agent currently cannot see its
    own prior-round artifacts at all.  This is a thin formatter over the
    unified :class:`ArtifactIndexView` — the projection (own vs dependency
    split, dedup, ACL grant) is computed once in :func:`build_artifact_index`,
    not re-derived here (shadow-implementation defect #5).
    """
    view = _artifact_view(state)
    if not view.artifacts:
        return ""
    # Own prior-round artifacts: full bodies so the agent resumes from them.
    return _render_artifact_index(
        view, budget=budget, body=True, only_own=True
    )


@user_block("evidence")
def _evidence(state, *, phase, budget, principal):
    ev = state.get("evidence", [])
    if not ev:
        return ""
    lines = [f"## Evidence ({len(ev)})"]
    for e in ev:
        loc = f"{e.get('file', '')}:{e.get('line', '')}" if e.get("file") else ""
        lines.append(f"- {loc}: {e.get('claim', '')}")
    return "\n".join(lines)


@user_block("report_chunks")
def _report_chunks(state, *, phase, budget, principal):
    chunks = state.get("report_chunks", [])
    if not chunks:
        return ""
    lines = [f"## Report Chunks ({len(chunks)})"]
    for c in chunks:
        lines.append(f"- {c}")
    return "\n".join(lines)


@user_block("module_inventory")
def _module_inventory(state, *, phase, budget, principal):
    inv = state.get("module_inventory", [])
    if not inv:
        return ""
    lines = [f"## Module Inventory ({len(inv)})"]
    for m in inv:
        if isinstance(m, dict):
            lines.append(f"- {m.get('module', '')} ({m.get('path', '')})")
        else:
            lines.append(f"- {m}")
    return "\n".join(lines)


@user_block("recovery_clues")
def _recovery_clues(state, *, phase, budget, principal):
    """Committed clues for the current step (blueprint §6 recovery journal).

    When a step is re-run after interruption, the agent must see what it
    already discovered — the clue chain is its surviving progress.  Rendered
    as a user block so retries resume from committed clues instead of
    re-exploring from scratch.
    """
    step = current_step(state)
    if not step:
        return ""
    step_id = step.get("step_id") or state.get("executing_step_id")
    if not step_id:
        return ""
    journal = (state.get("recovery") or {}).get(step_id) or {}
    clues = journal.get("clues") or []
    if not clues:
        return ""
    lines = [f"## Committed Clues ({len(clues)})"]
    lines.append("Findings already persisted for this step — resume from these, do NOT re-explore:")
    for c in clues:
        if isinstance(c, dict) and c.get("text"):
            lines.append(f"- {c['text']}")
    return "\n".join(lines)


@user_block("artifacts")
def _artifacts(state, *, phase, budget, principal):
    arts = state.get("artifacts", [])
    if not arts:
        return ""
    lines = [f"## Artifacts ({len(arts)})"]
    for a in arts:
        summary = a.get("summary", "")[:120]
        lines.append(f"- {a.get('artifact_id', '')} (kind={a.get('kind', '')}): {summary}")
    return "\n".join(lines)


@user_block("steering_artifacts")
def _steering_artifacts(state, *, phase, budget, principal):
    """Full artifact bodies for the steering act (own + dependency artifacts).

    The steering act must reason over artifact CONTENT, not just count or
    truncated summaries — a count-only view ("2 artifacts >= minimum, so the
    survey appears complete") produced premature SIGNAL proposals (bugreport
    -artifact v20260906).  The block carries the FULL body text of the
    executing step's own artifacts plus its access='hints' dependency
    artifacts — a thin formatter over the unified ArtifactIndexView, so the
    own/deps split and ACL grants are computed once (defect #5).
    """
    view = _artifact_view(state)
    if not view.artifacts:
        return ""
    return _render_artifact_index(
        view, budget=budget, body=True, only_own=False,
    )


def _artifact_view(state: dict[str, Any]) -> ArtifactIndexView:
    """The unified artifact projection for *state*.

    The canonical state-derived principal is used — the current
    step-instance is the sole authority on what it sees, so the renderer
    does not need the (possibly non-step) ``principal`` string the
    coordinator threaded through.  This is the single projection all three
    artifact renderers now consume.
    """
    return build_artifact_index(state, principal_from_state(state), acl_for_state(state))


def _render_artifact_index(
    view: ArtifactIndexView,
    budget: int = 0,
    body: bool = False,
    only_own: bool = False,
    self_teach: bool = True,
) -> str:
    """Canonical single renderer consumed by every artifact block.

    Formats the entries of the unified :class:`ArtifactIndexView` — never the
    raw artifacts again.  The own/dependency split is visible in the heading
    (``N own, M dependency (access=hints)``) and per-entry ``[own]`` /
    ``[dep:<owner>]`` tags, so the reader still knows which bodies are
    upstream — the contract the steering act relies on.

    ``body=True`` appends each artifact's full text and evidence (the
    steering act); ``body=False`` shows a compact id/kind/summary line (the
    StepAgent-facing view).  ``only_own`` restricts to this step's own
    artifacts (``own_history``); ``only_own=False`` shows own + dependency
    artifacts (``steering_artifacts``).  ``self_teach`` appends the
    resolvability sentence that teaches the model to treat ``artifact_id`` as
    the unit of continuity.
    """
    entries = view.own if only_own else view.artifacts
    if not entries:
        return ""

    n_own = len(view.own)
    n_dep = len(view.dependency)
    if only_own:
        heading = f"## Artifact Index — {n_own} own (this step)"
    else:
        heading = (
            f"## Artifact Index — {n_own} own, {n_dep} dependency (access=hints)"
        )
    lines = [heading]
    for e in entries:
        if e.step_id == view.principal_id:
            tag = "own"
        else:
            tag = f"dep:{e.step_id}"
        header = f"- {e.id} [{tag}] kind={e.kind}:"
        if e.summary:
            header += f" {e.summary[:160]}"
        lines.append(header)
        if body and e.body:
            lines.append(e.body)
            if e.evidences:
                lines.append("  Evidences:")
                lines.extend(f"  - {x}" for x in e.evidences)
    if self_teach:
        lines.append(
            "Full content: get_artifact(artifact_id). Artifacts NOT listed here "
            "are not currently readable by this step (declare depends_on + "
            'access="hints" to request them).'
        )
    text = "\n".join(lines)
    if budget:
        text = text[:budget].rstrip() + "\n\n---\n**CONTEXT TRUNCATED**"
    return text
