"""Algorithm layer — pure, zero-dependency computation.

All modules here are:
- **Pure functions** — deterministic, no IO, no side effects.
- **Zero-dependency** — stdlib only, no quro imports.
- **Independently testable** — unit-test without any setup.
"""

from quro.algorithm.goal_status import (
    Disposition,
    UNSATLevel,
    classify_unsat,
    compute_goal_status,
)
from quro.algorithm.signature import ProblemSignature

__all__ = [
    "Disposition",
    "ProblemSignature",
    "UNSATLevel",
    "classify_unsat",
    "compute_goal_status",
]
