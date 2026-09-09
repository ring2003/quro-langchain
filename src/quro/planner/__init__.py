"""Planner layer: planning bridge, scratchpad, and the meta-planner loop.

Provides:

- ``PlannerDomain`` — wraps quro-thinking's ``ExecutionPlanningDomain`` for
  ``StepType`` → ``PrimitiveOperator`` mapping and PEF-to-``Pipeline``
  conversion (the HTN ``g : Problem -> [StepSpec]``).
- ``Scratchpad`` / ``PPF`` / ``RoundRecord`` — the meta-planner's stateful
  planning surface (data model per ``meta-plan-prompt-and-ppf.md`` §2).
- ``Session`` — the framework-side single-round primitive (``f`` execution +
  fact-layer ``evaluate`` + ``report``).
- ``MetaPlannerLoop`` — the agent-driven scratchpad loop (outer ``g -> f``).
- ``goal_status`` — WorldState abstraction, UNSAT classification, and fact
  extraction (the framework fact layer).
"""

from quro.planner.domain import PlannerDomain
from quro.planner.goal_status import (
    AuditorFactExtractor,
    Disposition,
    Fact,
    FactExtractor,
    FactSource,
    ProblemSignature,
    RoundRecord as MemoryRoundRecord,
    UNSATDiagnosis,
    UNSATLevel,
    WorldState,
    classify_unsat,
    compute_goal_status,
    goal_status_from_results,
)
from quro.planner.meta_tools import (
    META_PLANNER_SYSTEM_PROMPT,
    make_meta_planner_tools,
)
from quro.planner.meta_prompt import MetaPlannerPromptManager
from quro.planner.meta_loop import (
    MetaPlannerLoop,
    MetaPlannerLoopConfig,
    MetaPlannerLoopResult,
)
from quro.planner.scratchpad import PPF, PlanGate, RoundRecord, Scratchpad
from quro.planner.session import Session, SessionResult

__all__ = [
    # domain
    "PlannerDomain",
    # scratchpad
    "Scratchpad",
    "PPF",
    "PlanGate",
    "RoundRecord",
    # session
    "Session",
    "SessionResult",
    # meta_tools
    "make_meta_planner_tools",
    "META_PLANNER_SYSTEM_PROMPT",
    # meta_prompt
    "MetaPlannerPromptManager",
    # meta_loop
    "MetaPlannerLoop",
    "MetaPlannerLoopConfig",
    "MetaPlannerLoopResult",
    # goal_status — world model
    "Fact",
    "FactSource",
    "FactExtractor",
    "AuditorFactExtractor",
    "WorldState",
    # goal_status — classification
    "Disposition",
    "UNSATLevel",
    "UNSATDiagnosis",
    "classify_unsat",
    "compute_goal_status",
    "goal_status_from_results",
    # goal_status — memory
    "ProblemSignature",
    "MemoryRoundRecord",
]
