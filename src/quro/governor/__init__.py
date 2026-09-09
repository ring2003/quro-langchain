"""Execution Governor — the single policy coordinator for step lifecycle.

The governor owns every decision about step termination, retry, rollback,
prompt assembly, and failure feedback.  ``StepAdapter`` and
``PipelineRunner`` delegate to it and become thin executors.

See ``docs/system-analysis/09-refactoring-implementation-plan.md`` §4.
"""

from quro.governor.core import (
    ExecutionGovernor,
    ProgressSnapshot,
    TerminationDecision,
    TerminationReason,
    TERMINAL_TOOLS,
)
from quro.governor.termination import DefaultTerminationPolicy, TerminationPolicy
from quro.governor.prompt import PromptAssemblyPolicy, DefaultPromptAssemblyPolicy
from quro.governor.feedback import FailureFeedback

__all__ = [
    "ExecutionGovernor",
    "ProgressSnapshot",
    "TerminationDecision",
    "TerminationReason",
    "TERMINAL_TOOLS",
    "DefaultTerminationPolicy",
    "TerminationPolicy",
    "PromptAssemblyPolicy",
    "DefaultPromptAssemblyPolicy",
    "FailureFeedback",
]
