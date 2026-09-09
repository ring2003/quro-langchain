"""Canonical ``phase → ops`` gating authority (Phase A).

This is the **single** ``PHASE_OPS`` table for the engineering workflow
domain.  Before Phase A the same semantic fact was written three times, with
drift:

- ``core/domain/workflow_domain.py`` — kernel gating authority (had
  ``carry_forward`` + ``request_clarification`` in ``step_execute``).
- ``core/phase_machine.py`` — legacy controller copy (drifted).
- a former ``roles/coordinator.py`` capability-shaped copy (different shape).

All three consumers now derive from this one table.  See
``docs/unsat-policy-loop/architecture/evolution.md`` §3 (Phase A).

This module is pure data + one helper: no runtime imports, no kernel deps.
"""

from __future__ import annotations

# phase → allowed domain ops.  Single gating authority for the domain
# (``EngineeringWorkflowDomain.apply``) and for the runtime's phase-aware
# tool resolution.
PHASE_OPS: dict[str, set[str]] = {
    "understanding": {"set_interpretation", "set_plan", "confirm_understanding", "carry_forward"},
    "step_spec": {
        "create_step", "update_step", "finalize_step",
        "resolve_clarification",
        "carry_forward",
    },
    "step_execute": {
        "add_artifact", "complete_step", "request_clarification",
        "terminate_subtree",
        "checkpoint", "backtrack",
        "get_artifact", "get_step",
        "carry_forward",
    },
    "step_execute:subtree": {
        "add_artifact", "complete_step", "terminate_subtree",
        "fold_tree",
        "checkpoint", "backtrack",
        "get_artifact", "get_step",
        "carry_forward",
    },
    "evaluate": {"submit_evaluation", "get_artifact", "get_step", "carry_forward"},
    "done": set(),
}

# Ops valid in every phase; OR'd into the gating set at runtime.
ALLOWED_IN_ANY_PHASE: frozenset[str] = frozenset({"checkpoint", "backtrack"})


def allowed_ops_for_phase(phase: str) -> set[str]:
    """Return the ops allowed in *phase*, including any-phase ops."""
    return PHASE_OPS.get(phase, set()) | set(ALLOWED_IN_ANY_PHASE)


# ---------------------------------------------------------------------------
# Terminal tool per phase (Phase 0 cleanup — sink from ``roles/coordinator.py``)
# ---------------------------------------------------------------------------

# Each phase is ended by ONE tool that advances the workflow forward.  A step
# whose session is in a given phase is terminated by that phase's terminal
# tool.  A ``StepType`` may override this via ``executor_hints["terminal_tool"]``
# (e.g. a step that spans understanding → step_spec ends at ``finalize_step``).
PHASE_TERMINAL_TOOL: dict[str, str] = {
    "understanding": "confirm_understanding",
    "step_spec": "finalize_step",
    "step_execute": "complete_step",
    "step_execute:subtree": "complete_step",
    "evaluate": "submit_evaluation",
}


def terminal_tool_for_phase(phase: str) -> str | None:
    """Return the terminal tool that ends *phase*, or ``None`` when unknown."""
    return PHASE_TERMINAL_TOOL.get(phase)
