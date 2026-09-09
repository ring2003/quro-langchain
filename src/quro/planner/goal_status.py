"""WorldState abstraction and UNSAT classification for the policy loop.

Bridges the gap between free-text pipeline artifacts and structured
``GoalStatus`` computation.  This is the evaluate-stage foundation for the
UNSAT-driven policy loop blueprint (§4 Evaluate stage, §9.1 world_state
extraction).

Architecture
------------

::

    PipelineResult ──FactExtractor──► WorldState ──compute_goal_status──► GoalStatus
                                          │
                                    classify_unsat()
                                          │
                                    UNSATDiagnosis
                                          │
                                    RoundRecord ──► compact / quro-memory

Two extraction strategies (blueprint §9.1):

1. **Rule-based** (``FactExtractor``) — reads ``StepResult.state`` keys as
   facts.  Fast, deterministic, zero LLM cost.  Best when step roles emit
   structured ``state`` dictionaries.

2. **Auditor-based** (``AuditorFactExtractor``) — LLM judges each goal fact
   against artifact summaries.  Handles free-text outputs but costs a
   (cheap) LLM call.  Recommended default for general roles.

See also
--------
- ``docs/unsat-driven-policy-loop-blueprint.md`` — full design
- ``quro_thinking.domains.planning.verification.compute_goal_status`` —
  upstream GoalStatus API that this module feeds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from quro.algorithm.goal_status import (
    Disposition,
    UNSATLevel,
    classify_unsat as _classify_unsat,
    compute_goal_status as _compute_goal_status,
)
from quro.algorithm.signature import ProblemSignature
from quro.pipeline.core import PipelineResult
from quro.steps.core import StepResult

# ============================================================================
# Fact & WorldState — the core world model
# ============================================================================


class FactSource(str, Enum):
    """Where a fact was observed or derived."""

    INITIAL = "initial"          # From goal_facts declaration
    STEP_STATE = "step_state"    # Extracted from StepResult.state dict
    ARTIFACT = "artifact"        # Extracted from artifact summary text
    AUDITOR = "auditor"          # Judged by LLM auditor role
    DERIVED = "derived"          # Inferred from other facts


@dataclass
class Fact:
    """A single observable proposition about the world.

    Mirrors ``quro_thinking.domains.planning.state.Fact`` but is
    self-contained so quro-langchain can operate without importing
    quro-thinking at the model level.

    Args:
        name: Fact identifier (e.g. ``"tests_pass"``, ``"code_compiles"``).
        value: The fact's current value — ``True``/``False`` for boolean
            goals, ``int``/``str`` for resource-type facts, ``None`` for
            unknown.
        source: Where this fact came from (``FactSource``).
        confidence: 0.0–1.0 confidence in this fact.  Step-state facts are
            1.0; auditor facts may be lower.
        sequence: Monotonically-increasing provenance marker.
        prior_value: Previous value, if this fact was overwritten.
    """

    name: str
    value: Any
    source: FactSource = FactSource.INITIAL
    confidence: float = 1.0
    sequence: int = 0
    prior_value: Any = None

    def is_true(self) -> bool:
        """Convenience: is this a satisfied boolean fact?"""
        return self.value is True

    def is_known(self) -> bool:
        """Is the value known (not None)?"""
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "sequence": self.sequence,
            "prior_value": self.prior_value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Fact:
        source_raw = d.get("source", "initial")
        source = FactSource(source_raw) if source_raw in FactSource._value2member_map_ else FactSource.INITIAL
        return cls(
            name=d["name"],
            value=d.get("value"),
            source=source,
            confidence=d.get("confidence", 1.0),
            sequence=d.get("sequence", 0),
            prior_value=d.get("prior_value"),
        )


@dataclass
class WorldState:
    """Snapshot of the observable world at a point in the policy loop.

    Compatible with ``quro_thinking.domains.planning.state.WorldState`` via
    ``to_quro_thinking()`` / ``from_quro_thinking()``.

    Args:
        facts: Named facts observed so far.
        resources: Named numeric resources (budgets, counts).
        temporal_state: Free-form temporal bookkeeping (step counts, etc.).
        metadata: Arbitrary annotations (decision graph, checkpoints, …).
    """

    facts: dict[str, Fact] = field(default_factory=dict)
    resources: dict[str, int | float] = field(default_factory=dict)
    temporal_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ access

    def get_fact(self, name: str) -> Fact | None:
        """Return the fact with *name*, or None."""
        return self.facts.get(name)

    def set_fact(self, fact: Fact) -> None:
        """Set (or overwrite) a fact, preserving prior_value."""
        old = self.facts.get(fact.name)
        if old is not None:
            fact.prior_value = old.value
        self.facts[fact.name] = fact

    def set_fact_value(
        self,
        name: str,
        value: Any,
        *,
        source: FactSource = FactSource.STEP_STATE,
        confidence: float = 1.0,
        sequence: int = 0,
    ) -> Fact:
        """Convenience: set a fact by name+value, return the Fact object."""
        f = Fact(
            name=name,
            value=value,
            source=source,
            confidence=confidence,
            sequence=sequence,
        )
        self.set_fact(f)
        return f

    def merge(self, other: WorldState) -> None:
        """Merge *other* facts/resources/temporal into this state.

        Existing facts are overwritten when *other* has a higher sequence.
        """
        for name, f in other.facts.items():
            existing = self.facts.get(name)
            if existing is None or f.sequence >= existing.sequence:
                self.set_fact(f)
        for k, v in other.resources.items():
            self.resources[k] = v
        self.temporal_state.update(other.temporal_state)
        self.metadata.update(other.metadata)

    def copy(self) -> WorldState:
        return WorldState(
            facts={k: Fact(**v.__dict__) for k, v in self.facts.items()},
            resources=dict(self.resources),
            temporal_state=dict(self.temporal_state),
            metadata=dict(self.metadata),
        )

    # -------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
            "resources": dict(self.resources),
            "temporal_state": dict(self.temporal_state),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldState:
        facts: dict[str, Fact] = {}
        for k, v in d.get("facts", {}).items():
            if isinstance(v, dict):
                facts[k] = Fact.from_dict(v)
            elif isinstance(v, Fact):
                facts[k] = v
        return cls(
            facts=facts,
            resources=dict(d.get("resources", {})),
            temporal_state=dict(d.get("temporal_state", {})),
            metadata=dict(d.get("metadata", {})),
        )

    # -------------------------------------------- quro-thinking compatibility

    def to_quro_thinking(self) -> Any:
        """Convert to ``quro_thinking.domains.planning.state.WorldState``.

        Returns a quro-thinking WorldState suitable for passing to
        ``compute_goal_status()`` / ``validate_unsat_proof()``.
        """
        try:
            from quro_thinking.domains.planning.state import (
                Fact as QTFact,
                WorldState as QTWorldState,
            )

            qt_facts: dict[str, Any] = {}
            for name, f in self.facts.items():
                qt_facts[name] = QTFact(
                    name=f.name,
                    value=f.value,
                    sequence=f.sequence,
                    prior_value=f.prior_value,
                )
            return QTWorldState(
                facts=qt_facts,
                resources=dict(self.resources),
                temporal_state=dict(self.temporal_state),
                metadata=dict(self.metadata),
            )
        except ImportError:
            raise ImportError(
                "quro-thinking is required for GoalStatus computation. "
                "Install it or use WorldState in standalone mode."
            )

    @classmethod
    def from_quro_thinking(cls, qt_ws: Any) -> WorldState:
        """Create from a quro-thinking WorldState."""
        facts: dict[str, Fact] = {}
        for name, qf in getattr(qt_ws, "facts", {}).items():
            facts[name] = Fact(
                name=getattr(qf, "name", name),
                value=getattr(qf, "value", None),
                source=FactSource.INITIAL,
                sequence=getattr(qf, "sequence", 0),
                prior_value=getattr(qf, "prior_value", None),
            )
        return cls(
            facts=facts,
            resources=dict(getattr(qt_ws, "resources", {})),
            temporal_state=dict(getattr(qt_ws, "temporal_state", {})),
            metadata=dict(getattr(qt_ws, "metadata", {})),
        )

    # ------------------------------------------------------------- summary

    def summary(self) -> str:
        """One-line summary for logging / compact injection."""
        sat = [n for n, f in self.facts.items() if f.is_true()]
        unsat = [n for n, f in self.facts.items() if f.is_known() and not f.is_true()]
        unknown = [n for n, f in self.facts.items() if not f.is_known()]
        parts: list[str] = []
        if sat:
            parts.append(f"SAT={sat}")
        if unsat:
            parts.append(f"UNSAT={unsat}")
        if unknown:
            parts.append(f"UNKNOWN={unknown}")
        parts.append(f"step={self.temporal_state.get('step_count', 0)}")
        return " ".join(parts)


# ============================================================================
# UNSAT classification — re-exported from algorithm layer
# ============================================================================

# UNSATLevel, Disposition are imported from quro.algorithm.goal_status above.
# They are re-exported here for backward compatibility.


@dataclass
class UNSATDiagnosis:
    """Structured diagnosis of *why* the policy loop didn't reach SAT.

    This is the output of the evaluate stage — it feeds both the compact
    stage (for memory) and the replan stage (for targeted disposition).

    Args:
        level: Which of the four UNSAT levels applies.
        unsat_facts: Specific fact names that are not satisfied.
        sat_facts: Fact names that *are* satisfied.
        cause: Human-readable cause of the failure.
        step_id: For RUN-UNSAT, the failing step ID.
        step_error: For RUN-UNSAT, the error message.
        proof_type: For PROOF-UNSAT, ``"resource_exhaustion"`` or ``"state_trap"``.
        exhausted_checkpoints: Breadcrumbs that were exhausted.
        disposition: Recommended next action.
        round_index: Which policy-loop round produced this diagnosis.
        stall_count: How many consecutive rounds with unchanged UNSAT_facts.
    """

    level: UNSATLevel
    unsat_facts: list[str] = field(default_factory=list)
    sat_facts: list[str] = field(default_factory=list)
    cause: str = ""
    step_id: str | None = None
    step_error: str | None = None
    proof_type: str | None = None
    exhausted_checkpoints: list[str] = field(default_factory=list)
    disposition: Disposition = Disposition.CONTINUE
    round_index: int = 0
    stall_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "unsat_facts": list(self.unsat_facts),
            "sat_facts": list(self.sat_facts),
            "cause": self.cause,
            "step_id": self.step_id,
            "step_error": self.step_error,
            "proof_type": self.proof_type,
            "exhausted_checkpoints": list(self.exhausted_checkpoints),
            "disposition": self.disposition.value,
            "round_index": self.round_index,
            "stall_count": self.stall_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UNSATDiagnosis:
        return cls(
            level=UNSATLevel(d["level"]),
            unsat_facts=list(d.get("unsat_facts", [])),
            sat_facts=list(d.get("sat_facts", [])),
            cause=d.get("cause", ""),
            step_id=d.get("step_id"),
            step_error=d.get("step_error"),
            proof_type=d.get("proof_type"),
            exhausted_checkpoints=list(d.get("exhausted_checkpoints", [])),
            disposition=Disposition(d.get("disposition", "continue")),
            round_index=d.get("round_index", 0),
            stall_count=d.get("stall_count", 0),
        )

    # -- convenience builders -----------------------------------------------

    @classmethod
    def sat(cls) -> UNSATDiagnosis:
        """No failure — everything is satisfied."""
        return cls(level=UNSATLevel.VERIFY_UNSAT, unsat_facts=[])  # unsat_facts empty → SAT

    @property
    def is_sat(self) -> bool:
        """True when all goals are met (no unsat_facts AND no step errors)."""
        return (
            len(self.unsat_facts) == 0
            and self.level != UNSATLevel.PLAN_UNSAT
            and self.level != UNSATLevel.RUN_UNSAT
            and self.level != UNSATLevel.PROOF_UNSAT
        )

    @property
    def is_terminal_unsat(self) -> bool:
        """True when we should stop retrying (PROOF-UNSAT or exhausted)."""
        return self.level == UNSATLevel.PROOF_UNSAT


# ============================================================================
# Fact extraction — bridge from pipeline output to WorldState (blueprint §9.1)
# ============================================================================


class IFactExtractor(Protocol):
    """Protocol for extracting facts from step/pipeline results."""

    def extract_from_step(self, result: StepResult) -> WorldState:
        """Extract facts from a single step result."""
        ...

    def extract_from_pipeline(self, result: PipelineResult) -> WorldState:
        """Extract facts from a completed pipeline result."""
        ...


class FactExtractor:
    """Rule-based fact extraction from ``StepResult.state`` dictionaries.

    This is the **deterministic** strategy (blueprint §9.1 option b).
    Steps that emit structured ``state`` dicts get zero-cost fact extraction.

    A step's ``state`` dict is treated as a flat fact set:

    - ``state["key"] = True/False`` → ``Fact(name=key, value=...)``
    - ``state["key"] = <number>`` → ``Fact(name=key, value=<number>)``
    - ``state["key"] = "str"`` → ``Fact(name=key, value="str")``

    Non-dict ``state`` values are ignored.
    """

    def __init__(self, *, fact_prefix: str = "") -> None:
        self._fact_prefix = fact_prefix

    def extract_from_step(self, result: StepResult) -> WorldState:
        """Extract facts from a single ``StepResult``.

        Returns a WorldState containing facts derived from:
        1. ``result.state`` — treated as a flat fact dict.
        2. ``result.ok`` → ``Fact(name="{step_id}_ok", value=ok)``.
        3. ``result.artifacts`` → ``Fact(name="{step_id}_artifact_count", value=n)``.
        """
        ws = WorldState()
        seq = 0

        # Step-level success fact.
        ws.set_fact(Fact(
            name=self._make_name(f"{result.step_id}_ok"),
            value=result.ok,
            source=FactSource.STEP_STATE,
            confidence=1.0,
            sequence=seq,
        ))
        seq += 1

        # Artifact count.
        ws.set_fact(Fact(
            name=self._make_name(f"{result.step_id}_artifact_count"),
            value=len(result.artifacts),
            source=FactSource.ARTIFACT,
            confidence=1.0,
            sequence=seq,
        ))
        seq += 1

        # State dict facts.
        if isinstance(result.state, dict):
            for key, value in result.state.items():
                ws.set_fact(Fact(
                    name=self._make_name(key),
                    value=value,
                    source=FactSource.STEP_STATE,
                    confidence=1.0,
                    sequence=seq,
                ))
                seq += 1

        # Artifact content hints.
        for i, artifact in enumerate(result.artifacts):
            if isinstance(artifact, dict):
                # Try common artifact keys.
                for hint_key in ("status", "ok", "result", "summary"):
                    if hint_key in artifact:
                        ws.set_fact(Fact(
                            name=self._make_name(f"{result.step_id}_artifact_{i}_{hint_key}"),
                            value=artifact[hint_key],
                            source=FactSource.ARTIFACT,
                            confidence=0.8,  # lower confidence for artifact hints
                            sequence=seq,
                        ))
                        seq += 1

        return ws

    def extract_from_pipeline(self, result: PipelineResult) -> WorldState:
        """Extract facts from a completed pipeline.

        Merges per-step facts and adds pipeline-level facts.
        """
        ws = WorldState()

        ws.set_fact(Fact(
            name=self._make_name("pipeline_ok"),
            value=result.ok,
            source=FactSource.DERIVED,
            confidence=1.0,
        ))
        ws.set_fact(Fact(
            name=self._make_name("pipeline_partial"),
            value=result.partial,
            source=FactSource.DERIVED,
            confidence=1.0,
        ))
        ws.temporal_state["step_count"] = len(result.order)

        for step_id, sr in result.step_results.items():
            step_ws = self.extract_from_step(sr)
            ws.merge(step_ws)

        if result.error:
            ws.metadata["pipeline_error"] = result.error
        if result.replan_reason:
            ws.metadata["replan_reason"] = result.replan_reason

        return ws

    def _make_name(self, name: str) -> str:
        if self._fact_prefix:
            return f"{self._fact_prefix}.{name}"
        return name


class AuditorFactExtractor:
    """LLM-based fact extraction (blueprint §9.1 option a).

    Delegates fact judgment to an auditor role (cheap LLM) that reads
    artifact summaries and judges each goal fact.  This is the recommended
    strategy for general roles whose output is free-text.

    **Not yet implemented** — placeholder for Phase 1+.  The ``extract``
    method signature is defined so callers can write against it.

    Args:
        backend: An LLM backend that supports ``complete(prompt) -> str``.
        model: Optional model override (default: same as evaluator model).
    """

    def __init__(self, backend: Any = None, *, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    def extract(
        self,
        goal_facts: list[str],
        artifact_summaries: list[str],
    ) -> WorldState:
        """Judge each goal fact against artifact summaries via LLM.

        Args:
            goal_facts: List of fact names to judge (e.g. ``["tests_pass",
                "code_compiles"]``).
            artifact_summaries: Summaries of pipeline step outputs.

        Returns:
            WorldState with one Fact per goal_fact, each with
            ``source=FactSource.AUDITOR`` and ``confidence`` from the LLM's
            expressed certainty.

        Raises:
            NotImplementedError: In this placeholder; will be implemented in
                Phase 1+ when the auditor role is wired.
        """
        raise NotImplementedError(
            "AuditorFactExtractor is a placeholder for Phase 1+. "
            "Use FactExtractor (rule-based) for now."
        )


# ============================================================================
# GoalStatus computation (blueprint §4 Evaluate stage)
# ============================================================================


def compute_goal_status(
    world_state: WorldState,
    goal_facts: list[str],
) -> UNSATDiagnosis:
    """Compute goal satisfaction from a WorldState.

    Delegates to ``quro.algorithm.goal_status.compute_goal_status`` — a
    pure function that operates on a plain dict.  We extract
    ``{name: is_true()}`` from the WorldState before calling.

    Args:
        world_state: The accumulated world state after pipeline execution.
        goal_facts: List of fact names that must all be ``True`` for SAT.

    Returns:
        ``UNSATDiagnosis`` with ``level=VERIFY_UNSAT`` and populated
        ``sat_facts`` / ``unsat_facts``.
    """
    # Extract plain dict from WorldState for the pure algorithm.
    plain_facts: dict[str, bool | None] = {}
    for name, f in world_state.facts.items():
        if f.is_known():
            plain_facts[name] = bool(f.value)
        else:
            plain_facts[name] = None

    sat, unsat, unknown = _compute_goal_status(plain_facts, goal_facts)

    cause_parts: list[str] = []
    if unsat:
        cause_parts.append(f"Unsatisfied: {unsat}")
    if unknown:
        cause_parts.append(f"Unknown: {unknown}")

    # When all goal_facts are satisfied (no unsat, no unknown), emit
    # a VERIFY_UNSAT-level diagnosis with empty unsat_facts — the
    # downstream ``is_sat`` property treats this as SAT.
    if not unsat and not unknown:
        return UNSATDiagnosis(
            level=UNSATLevel.VERIFY_UNSAT,
            sat_facts=sat,
            unsat_facts=[],
            cause="All goals satisfied — pipeline completed successfully",
        )

    return UNSATDiagnosis(
        level=UNSATLevel.VERIFY_UNSAT,
        sat_facts=sat,
        unsat_facts=unsat,
        cause="; ".join(cause_parts) if cause_parts else "All goals satisfied",
    )


def classify_unsat(
    diagnosis: UNSATDiagnosis,
    *,
    pef_empty: bool = False,
    step_error: str | None = None,
    step_id: str | None = None,
    proof_evidence: dict[str, Any] | None = None,
    exhausted_checkpoints: list[str] | None = None,
    previous_unsat_facts: list[str] | None = None,
    round_index: int = 0,
    max_stall_rounds: int = 2,
) -> UNSATDiagnosis:
    """Classify failure into the correct UNSAT level and disposition.

    Delegates to ``quro.algorithm.goal_status.classify_unsat`` — a pure
    function with no quro-langchain dependencies.
    """
    proof_type = proof_evidence.get("type") if proof_evidence else None

    raw = _classify_unsat(
        unsat_facts=diagnosis.unsat_facts,
        sat_facts=diagnosis.sat_facts,
        pef_empty=pef_empty,
        step_error=step_error or diagnosis.step_error,
        step_id=step_id or diagnosis.step_id,
        proof_type=proof_type,
        previous_unsat_facts=previous_unsat_facts,
        round_index=round_index,
        max_stall_rounds=max_stall_rounds,
    )

    return UNSATDiagnosis(
        level=UNSATLevel(raw["level"]),
        unsat_facts=raw["unsat_facts"],
        sat_facts=raw["sat_facts"],
        cause=raw["cause"],
        step_id=raw.get("step_id"),
        step_error=raw.get("step_error"),
        proof_type=raw.get("proof_type"),
        exhausted_checkpoints=list(exhausted_checkpoints or diagnosis.exhausted_checkpoints),
        disposition=Disposition(raw["disposition"]),
        round_index=round_index,
        stall_count=raw.get("stall_count", 0),
    )


# ============================================================================
# Main entry point — evaluate stage (blueprint §4)
# ============================================================================


def goal_status_from_results(
    pipeline_result: PipelineResult,
    goal_facts: list[str],
    *,
    extractor: IFactExtractor | None = None,
    pef_empty: bool = False,
    proof_evidence: dict[str, Any] | None = None,
    exhausted_checkpoints: list[str] | None = None,
    previous_unsat_facts: list[str] | None = None,
    round_index: int = 0,
    max_stall_rounds: int = 2,
) -> UNSATDiagnosis:
    """Evaluate pipeline results against goal facts — the evaluate stage.

    This is the main function called by ``Session.evaluate`` after each
    pipeline execution (blueprint §4 Evaluate stage).

    Flow::

        1. Extract WorldState from PipelineResult (FactExtractor)
        2. compute_goal_status(world_state, goal_facts)
        3. classify_unsat(diagnosis, ...) — determine level + disposition

    Args:
        pipeline_result: Result from ``PipelineRunner.run()``.
        goal_facts: Fact names that must be ``True`` for success.
        extractor: Fact extraction strategy (default: ``FactExtractor``).
        pef_empty: Whether the solver produced an empty PEF.
        proof_evidence: UNSAT proof from quro-thinking, if any.
        exhausted_checkpoints: Solver's exhausted branch labels.
        previous_unsat_facts: UNSAT_facts from the prior round.
        round_index: Current policy-loop round number.
        max_stall_rounds: Stalled rounds before escalation.

    Returns:
        ``UNSATDiagnosis`` with correct level, facts, and disposition.
    """
    ext = extractor or FactExtractor()

    # Check for step-level failures first (RUN-UNSAT candidates).
    step_error: str | None = None
    step_id: str | None = None
    for sid, sr in pipeline_result.step_results.items():
        if not sr.ok:
            step_error = sr.error or "unknown error"
            step_id = sid
            break

    # Extract world state from results.
    world_state = ext.extract_from_pipeline(pipeline_result)

    # Auto-resolve unknown goal facts when all steps succeeded AND
    # produced material output (artifacts or non-trivial state).
    # Steps emit structured state keys (e.g. "t_xxx_ok") but never
    # explicitly set high-level goal fact names like "app_built" or
    # "tests_pass".  When every step is OK, the goal fact is absent
    # from the world state, AND steps produced evidence (artifacts),
    # infer it as SAT — the pipeline completed successfully, so its
    # outputs satisfy the goals.
    all_steps_ok = (
        all(sr.ok for sr in pipeline_result.step_results.values())
        if pipeline_result.step_results else False
    )
    if all_steps_ok and goal_facts:
        has_evidence = any(
            sr.artifacts or (isinstance(sr.state, dict) and len(sr.state) > 1)
            for sr in pipeline_result.step_results.values()
        )
        if has_evidence:
            for gf in goal_facts:
                if gf not in world_state.facts:
                    world_state.set_fact(Fact(
                        name=gf,
                        value=True,
                        source=FactSource.DERIVED,
                        confidence=0.9,
                        sequence=999,
                    ))

    # Compute goal satisfaction.
    base_diagnosis = compute_goal_status(world_state, goal_facts)

    # Set extra metadata.
    world_state.metadata.setdefault("pipeline_error", pipeline_result.error)
    world_state.metadata.setdefault("replan_reason", pipeline_result.replan_reason)
    world_state.metadata["round_index"] = round_index
    base_diagnosis.exhausted_checkpoints = list(exhausted_checkpoints or [])

    # Classify.
    return classify_unsat(
        base_diagnosis,
        pef_empty=pef_empty,
        step_error=step_error,
        step_id=step_id,
        proof_evidence=proof_evidence,
        exhausted_checkpoints=exhausted_checkpoints,
        previous_unsat_facts=previous_unsat_facts,
        round_index=round_index,
        max_stall_rounds=max_stall_rounds,
    )


# ============================================================================
# FileArtifactChecker — round-goal artifact/file acceptance (demo7 §4.4)
# ============================================================================


class FileArtifactChecker:
    """Deterministic predicate for round-goal artifact/file acceptance.

    demo7 round goals are "must write report to <path>" — a file-existence /
    artifact check, not a business-level fact.  This checker verifies whether
    each round-goal fact (a file name or path) was produced as a
    ``kind="file"`` artifact (or exists on disk), so the round can be
    declared "done enough" deterministically — keeping the fact layer
    (``goal_status``) separate from the agent's own conclusion.
    """

    def check(
        self,
        round_goal_facts: list[str],
        pipeline_result: PipelineResult,
    ) -> tuple[list[str], list[str]]:
        """Return ``(sat_facts, unsat_facts)`` for the round goal."""
        produced_paths: set[str] = set()
        for sr in pipeline_result.step_results.values():
            for art in getattr(sr, "artifacts", []) or []:
                if not isinstance(art, dict):
                    continue
                if art.get("kind") == "file":
                    body = str(art.get("body", "") or "")
                    if body:
                        produced_paths.add(body)

        sat: list[str] = []
        unsat: list[str] = []
        for fact in round_goal_facts:
            if self._is_satisfied(fact, produced_paths):
                sat.append(fact)
            else:
                unsat.append(fact)
        return sat, unsat

    @staticmethod
    def _is_satisfied(fact: str, produced_paths: set[str]) -> bool:
        if not fact:
            return False
        # Direct match on the artifact body path.
        if fact in produced_paths:
            return True
        # Match on basename (e.g. "report-1.md" vs "/abs/.../report-1.md").
        if any(Path(p).name == fact for p in produced_paths):
            return True
        # Fall back to on-disk existence (absolute or relative path).
        try:
            if Path(fact).expanduser().is_file():
                return True
        except OSError:
            pass
        return False


# ============================================================================
# ProblemSignature & RoundRecord — compact / distillation (blueprint §6–7)
# ============================================================================


# ProblemSignature is imported from quro.algorithm.signature above.


@dataclass
class RoundRecord:
    """Per-round distillation record for the compact stage (blueprint §6).

    Written to quro-memory at L1 as ``type="fact"`` with idempotent
    ``memory_id``.  Contains enough information for cross-project recall
    (L2) to surface historical solutions / known dead ends.

    Args:
        signature: Stable problem identity.
        round_index: 0-based round number.
        diagnosis: The UNSAT diagnosis for this round.
        world_state_summary: Condensed world state snapshot.
        completed_step_ids: Steps that finished this round.
        pipeline_error: Pipeline-level error, if any.
    """

    signature: ProblemSignature
    round_index: int
    diagnosis: UNSATDiagnosis
    world_state_summary: str = ""
    completed_step_ids: list[str] = field(default_factory=list)
    pipeline_error: str | None = None

    @property
    def memory_id(self) -> str:
        """Idempotent memory id for quro-memory."""
        return self.signature.to_memory_id(self.round_index)

    def to_memory_body(self) -> str:
        """Render as a quro-memory ``type="fact"`` body (blueprint §6 L1)."""
        parts: list[str] = []
        parts.append(f"Round {self.round_index}:")
        parts.append(f"  Level: {self.diagnosis.level.value}")
        parts.append(f"  SAT_facts: {self.diagnosis.sat_facts}")
        parts.append(f"  UNSAT_facts: {self.diagnosis.unsat_facts}")
        parts.append(f"  Cause: {self.diagnosis.cause}")
        parts.append(f"  Disposition: {self.diagnosis.disposition.value}")
        if self.completed_step_ids:
            parts.append(f"  Completed: {self.completed_step_ids}")
        if self.pipeline_error:
            parts.append(f"  PipelineError: {self.pipeline_error}")
        if self.world_state_summary:
            parts.append(f"  State: {self.world_state_summary}")
        return "\n".join(parts)

    def to_memory_tags(self) -> list[str]:
        """Tags for quro-memory."""
        tags = self.signature.to_memory_tags()
        tags.append(f"round:{self.round_index}")
        tags.append("unsat")
        if self.diagnosis.level == UNSATLevel.PROOF_UNSAT:
            tags.append("pattern:proof_unsat")
        elif self.diagnosis.level == UNSATLevel.PLAN_UNSAT:
            tags.append("pattern:htn_unsolvable")
        elif self.diagnosis.unsat_facts:
            tags.append("pattern:goal_unmet")
        return tags

    def to_dict(self) -> dict[str, Any]:
        return {
            "sig": self.signature.sig,
            "round_index": self.round_index,
            "diagnosis": self.diagnosis.to_dict(),
            "world_state_summary": self.world_state_summary,
            "completed_step_ids": list(self.completed_step_ids),
            "pipeline_error": self.pipeline_error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoundRecord:
        return cls(
            signature=ProblemSignature(
                sig=d["sig"],
                goal_facts=(),
                problem_prefix="",
            ),
            round_index=d["round_index"],
            diagnosis=UNSATDiagnosis.from_dict(d["diagnosis"]),
            world_state_summary=d.get("world_state_summary", ""),
            completed_step_ids=list(d.get("completed_step_ids", [])),
            pipeline_error=d.get("pipeline_error"),
        )
