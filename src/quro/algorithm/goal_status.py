"""Algorithm layer — pure functions for goal-status computation and UNSAT
classification.

These functions are **zero-dependency** (stdlib only), deterministic, and
independently testable.  They accept plain dataclass-like inputs so callers
can use any WorldState / Fact representation, not just quro-langchain's own
types.

See also
--------
- ``docs/unsat-driven-policy-loop-blueprint.md`` §3 (four UNSAT levels)
- ``quro_thinking.domains.planning.verification.compute_goal_status`` —
  upstream reference for the fact-checking logic
"""

from __future__ import annotations

from enum import Enum


# ===========================================================================
# Input types — protocol-compatible, no coupling to quro types
# ===========================================================================


class UNSATLevel(str, Enum):
    """The four failure levels from blueprint §3."""

    PLAN_UNSAT = "PLAN-UNSAT"
    """HTN solver cannot reduce the goal → empty PEF."""

    RUN_UNSAT = "RUN-UNSAT"
    """A step failed during execution → StepResult.ok=False."""

    VERIFY_UNSAT = "VERIFY-UNSAT"
    """Pipeline finished but goals unmet → GoalStatus.UNSAT_facts non-empty."""

    PROOF_UNSAT = "PROOF-UNSAT"
    """Proven genuinely unsolvable → RESOURCE_EXHAUSTION / STATE_TRAP."""


class Disposition(str, Enum):
    """What to do after an UNSAT diagnosis."""

    CONTINUE = "continue"
    """Deterministic replan (L0)."""

    RETRY_STEP = "retry_step"
    """Targeted retry of a specific step."""

    ADD_STEPS = "add_steps"
    """Replan with new constraints from UNSAT_facts."""

    ESCALATE_L1 = "escalate_l1"
    """Let LLM define_method (blueprint L1)."""

    ESCALATE_L2 = "escalate_l2"
    """Let LLM full rebuild (blueprint L2)."""

    ABORT = "abort"
    """Terminal — archive the failure pattern (PROOF-UNSAT)."""


# ===========================================================================
# compute_goal_status — pure fact-checking
# ===========================================================================


def compute_goal_status(
    facts: dict[str, bool | None],
    goal_facts: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Check which goal facts are SAT / UNSAT / UNKNOWN.

    This is the pure-algorithm core, decoupled from any WorldState or Fact
    class.  Callers map their state representation to a simple
    ``dict[str, bool|None]`` where:

    - ``True``  → fact is satisfied
    - ``False`` → fact is known but not satisfied
    - ``None``  → fact is unknown (treated as UNSAT)

    Args:
        facts: Mapping of fact name → truth value.
        goal_facts: Fact names that must all be ``True`` for SAT.

    Returns:
        ``(sat_facts, unsat_facts, unknown_facts)`` — three sorted lists.
        If ``unsat_facts`` is empty, all goals are SAT.
    """
    sat: list[str] = []
    unsat: list[str] = []
    unknown: list[str] = []

    for name in goal_facts:
        val = facts.get(name)
        if val is None:
            unknown.append(name)
            unsat.append(name)       # unknown → treated as unsatisfied
        elif val is True:
            sat.append(name)
        else:
            unsat.append(name)

    return (sat, unsat, unknown)


# ===========================================================================
# classify_unsat — deterministic escalation-ladder decision (blueprint §5)
# ===========================================================================


def classify_unsat(
    *,
    unsat_facts: list[str] | None = None,
    sat_facts: list[str] | None = None,
    pef_empty: bool = False,
    step_error: str | None = None,
    step_id: str | None = None,
    proof_type: str | None = None,
    previous_unsat_facts: list[str] | None = None,
    round_index: int = 0,
    max_stall_rounds: int = 2,
) -> dict:
    """Determine UNSAT level and disposition from raw signals.

    Pure function — no side effects, no IO.  Priority order (first
    match wins):

    1. PROOF-UNSAT — *proof_type* is non-None.
    2. PLAN-UNSAT  — *pef_empty* is True.
    3. RUN-UNSAT   — *step_error* is non-None.
    4. VERIFY-UNSAT — *unsat_facts* is non-empty.
    5. SAT          — everything else.

    Args:
        unsat_facts: Names of unsatisfied goal facts.
        sat_facts: Names of satisfied goal facts.
        pef_empty: Whether HTN solver returned empty PEF.
        step_error: Error from a failed step.
        step_id: ID of the failing step.
        proof_type: ``"resource_exhaustion"`` or ``"state_trap"``.
        previous_unsat_facts: UNSAT_facts from previous round (for stall).
        round_index: Current loop round (0-based).
        max_stall_rounds: Consecutive unchanged rounds before escalation.

    Returns:
        Dict with keys ``level``, ``disposition``, ``unsat_facts``,
        ``sat_facts``, ``cause``, ``step_id``, ``step_error``,
        ``proof_type``, ``stall_count``.
    """
    _usat = list(unsat_facts or [])
    _sat = list(sat_facts or [])

    # 1. PROOF-UNSAT
    if proof_type:
        return {
            "level": UNSATLevel.PROOF_UNSAT,
            "disposition": Disposition.ABORT,
            "unsat_facts": _usat,
            "sat_facts": _sat,
            "cause": f"PROOF-UNSAT: {proof_type}",
            "step_id": step_id,
            "step_error": step_error,
            "proof_type": proof_type,
            "stall_count": 0,
        }

    # 2. PLAN-UNSAT
    if pef_empty:
        disp = Disposition.ESCALATE_L1 if round_index > 0 else Disposition.CONTINUE
        return {
            "level": UNSATLevel.PLAN_UNSAT,
            "disposition": disp,
            "unsat_facts": _usat,
            "sat_facts": _sat,
            "cause": "PLAN-UNSAT: HTN solver returned an empty PEF",
            "step_id": step_id,
            "step_error": step_error,
            "proof_type": proof_type,
            "stall_count": 0,
        }

    # 3. RUN-UNSAT
    if step_error:
        return {
            "level": UNSATLevel.RUN_UNSAT,
            "disposition": Disposition.RETRY_STEP,
            "unsat_facts": _usat,
            "sat_facts": _sat,
            "cause": f"RUN-UNSAT: step '{step_id}' failed — {step_error}",
            "step_id": step_id,
            "step_error": step_error,
            "proof_type": proof_type,
            "stall_count": 0,
        }

    # 4. VERIFY-UNSAT
    if _usat:
        stall_count = 0
        cause_parts = [f"Unsatisfied: {_usat}"]

        # Stall detection.
        if previous_unsat_facts is not None:
            if set(previous_unsat_facts) == set(_usat):
                stall_count = 1  # caller accumulates across rounds
                cause_parts.append(
                    f"stalled (same UNSAT_facts as round {round_index - 1})"
                )

        disp = (
            Disposition.ESCALATE_L1
            if stall_count >= max_stall_rounds
            else Disposition.ADD_STEPS
        )

        return {
            "level": UNSATLevel.VERIFY_UNSAT,
            "disposition": disp,
            "unsat_facts": _usat,
            "sat_facts": _sat,
            "cause": "; ".join(cause_parts),
            "step_id": step_id,
            "step_error": step_error,
            "proof_type": proof_type,
            "stall_count": stall_count,
        }

    # 5. SAT
    return {
        "level": UNSATLevel.VERIFY_UNSAT,
        "disposition": Disposition.CONTINUE,
        "unsat_facts": [],
        "sat_facts": _sat,
        "cause": "All goals satisfied",
        "step_id": step_id,
        "step_error": step_error,
        "proof_type": proof_type,
        "stall_count": 0,
    }
