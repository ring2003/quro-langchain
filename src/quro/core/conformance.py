"""ConformanceChecker — IConformanceChecker implementation.

Two static checks:
- check_wiring: run once at Session construction.
- check_round_batch: run before IPipelineController.execute().

Phase 5 deliverable (architecture §3.7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from quro.core.protocols import (
    IAuditLog,
    IConformanceChecker,
    IStepBuilder,
    Violation,
)

# An artifact id literal (unambiguous — unlike a step-name substring).
_ARTIFACT_ID_RE = re.compile(r"art_[0-9a-f]{8}")


@dataclass
class SessionDependencies:
    """All injected dependencies for a Session — checked by check_wiring."""

    builder: Any  # IStepBuilder
    store: Any  # IResourceStore
    audit_log: Any  # IAuditLog
    controller: Any  # IPipelineController
    handoff: Any | None = None  # ICrossRoundHandoff


def _is_in_memory_store(store: Any) -> bool:
    """Check if a store resolves to an in-memory backend."""
    store_type = type(store).__name__
    return "InMemory" in store_type or "Memory" in store_type


class ConformanceChecker:
    """IConformanceChecker implementation.

    check_wiring: run once at Session construction.
    check_round_batch: run before IPipelineController.execute().
    """

    def __init__(
        self,
        testing: bool = False,
        step_types: Any | None = None,
    ) -> None:
        self._testing = testing
        self._step_types = step_types

    def check_wiring(self, session_deps: SessionDependencies) -> list[Violation]:
        """Check wiring conformance at Session construction time."""
        violations: list[Violation] = []

        # 1. Assert IStepBuilder protocol.
        if not isinstance(session_deps.builder, IStepBuilder):
            violations.append(Violation(
                kind="missing_protocol",
                component="builder",
                message="builder does not implement IStepBuilder",
            ))

        # 2. Assert IResourceStore is not in-memory outside testing.
        if not self._testing and _is_in_memory_store(session_deps.store):
            violations.append(Violation(
                kind="in_memory_store",
                component="store",
                message="IResourceStore resolves to in-memory backend outside testing=True",
            ))

        # 3. Assert IAuditLog.
        if not isinstance(session_deps.audit_log, IAuditLog):
            violations.append(Violation(
                kind="missing_protocol",
                component="audit_log",
                message="audit_log does not implement IAuditLog",
            ))

        return violations

    def check_round_batch(self, batch: Any) -> list[Violation]:
        """Check round-batch conformance before pipeline execute."""
        violations: list[Violation] = []

        if self._step_types is None:
            return violations

        # Heuristic: flag a step whose objective references another step's
        # output in free text without a matching depends_on.
        if not hasattr(batch, "pending_steps"):
            return violations

        known_ids = set(batch.pending_steps.keys())
        for step_id, result in batch.pending_steps.items():
            obj = result.spec.objective
            for other_id in known_ids:
                if other_id != step_id and other_id in obj:
                    if other_id not in result.spec.depends_on:
                        violations.append(Violation(
                            kind="implicit_dependency",
                            component=step_id,
                            message=(
                                f"step '{step_id}' objective references "
                                f"'{other_id}' without depends_on"
                            ),
                        ))

        # Anti-Pattern I (unreachable artifact ref): if a step's objective /
        # expected_output names a prior artifact by its id (art_<0-9a-f>{8} —
        # unambiguous, unlike a step-name substring) but declares NO depends_on,
        # the referenced artifact can never enter the carried state, so the
        # instruction is meaningless to the executor.  Flag it (low false-
        # positive: referencing prior work without declaring any dependency).
        # We cannot map an artifact id to its owning step at planning time, so
        # we only assert the *necessary* condition (some depends_on declared);
        # the exact-owner check is the planner's own rule (meta_prompt rule 4).
        text = " ".join(filter(None, [
            getattr(result.spec, "objective", "") or "",
            getattr(result.spec, "expected_output", "") or "",
        ]))
        dep_ids = set(result.spec.depends_on or [])
        if text and not dep_ids:
            for aid in _ARTIFACT_ID_RE.findall(text):
                violations.append(Violation(
                    kind="unreachable_artifact_ref",
                    component=step_id,
                    message=(
                        f"step '{step_id}' references artifact '{aid}' in its "
                        f"objective/expected_output but declares no depends_on; "
                        f"declare the edge + access=\"hints\" so the artifact "
                        f"enters carried state, or the reference is unreachable "
                        f"(Anti-Pattern I)."
                    ),
                ))

        return violations
