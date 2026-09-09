"""RoundScope — one round's mutable step collection.

Replaces the closure-captured ``_manual_steps`` in ``make_meta_planner_tools``.
Constructed fresh at round open, discarded at round close.

Phase 3 deliverable (architecture §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quro.core.protocols import StepBuildResult
from quro.steps.core import StepSpec


@dataclass
class RoundScope:
    """One round's mutable step collection — replaces _manual_steps closure.

    Constructed fresh at round open, discarded at round close.
    """

    round_idx: int
    pending_steps: dict[str, StepBuildResult] = field(default_factory=dict)
    executed: bool = False

    def add(self, result: StepBuildResult) -> None:
        self.pending_steps[result.spec.id] = result

    def has(self, step_id: str) -> bool:
        return step_id in self.pending_steps

    def get(self, step_id: str) -> StepBuildResult | None:
        return self.pending_steps.get(step_id)

    def topological_batch(self) -> list[StepSpec]:
        """Return pending steps in dependency order."""
        if not self.pending_steps:
            return []

        # Build dependency graph.
        id_to_result = {sid: r for sid, r in self.pending_steps.items()}
        in_degree: dict[str, int] = {sid: 0 for sid in id_to_result}
        dependents: dict[str, list[str]] = {sid: [] for sid in id_to_result}

        for sid, result in id_to_result.items():
            for dep in result.spec.depends_on:
                if dep in id_to_result:
                    in_degree[sid] += 1
                    dependents[dep].append(sid)

        # Topological sort (Kahn's algorithm).
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result: list[StepSpec] = []

        while queue:
            # Sort for deterministic output.
            queue.sort()
            sid = queue.pop(0)
            result.append(id_to_result[sid].spec)
            for dep_sid in dependents[sid]:
                in_degree[dep_sid] -= 1
                if in_degree[dep_sid] == 0:
                    queue.append(dep_sid)

        return result

    def unexecuted_count(self) -> int:
        return len(self.pending_steps) if not self.executed else 0
