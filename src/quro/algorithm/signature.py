"""ProblemSignature — stable, hashable problem identity.

Pure computation, zero dependencies beyond stdlib.
Used for memory recall across rounds and projects (blueprint §7.1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemSignature:
    """Stable problem identity for memory recall.

    Computed as ``sha1(canonicalize(goal_facts) + first N chars of problem)``.
    Frozen dataclass — safe as dict key or set member.

    Args:
        sig: 16-char hex digest.
        goal_facts: Sorted tuple of goal fact names.
        problem_prefix: First ``prefix_chars`` of the problem string.
    """

    sig: str
    goal_facts: tuple[str, ...]
    problem_prefix: str

    @classmethod
    def compute(
        cls,
        problem: str,
        goal_facts: list[str],
        *,
        prefix_chars: int = 200,
    ) -> ProblemSignature:
        """Compute a deterministic signature.

        Goal facts are sorted before hashing so order is irrelevant.
        """
        sorted_facts = tuple(sorted(goal_facts))
        canonical = json.dumps(sorted_facts, sort_keys=True)
        prefix = problem[:prefix_chars]
        payload = canonical + prefix
        sig = hashlib.sha1(payload.encode()).hexdigest()[:16]
        return cls(sig=sig, goal_facts=sorted_facts, problem_prefix=prefix)

    def __str__(self) -> str:
        return self.sig

    def to_memory_id(self, round_index: int) -> str:
        """Idempotent memory id for storage backends."""
        return f"{self.sig}:r{round_index}"

    def to_memory_tags(self) -> list[str]:
        """Default tags for storage backends."""
        return [f"problem:{self.sig}"]
