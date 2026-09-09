"""Audit log hook — logs step entry/exit with timing."""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

from quro.steps.hooks import HookContext

if TYPE_CHECKING:
    from quro.steps.core import StepResult, StepSpec

logger = logging.getLogger(__name__)


class AuditLogHook:
    """Logs step start/end with timing for observability.

    Usage in YAML::

        pre_hooks:
          - name: audit_log
        post_hooks:
          - name: audit_log
    """

    name: str = "audit_log"

    def on_pre_step(
        self, step: StepSpec, context: HookContext
    ) -> None:
        context.metadata["_audit_start"] = time.time()
        logger.info(
            "Step '%s' starting (step_type=%s, objective=%.80s...)",
            step.id, step.step_type, step.objective,
        )

    def on_post_step(
        self,
        step: StepSpec,
        result: StepResult,
        context: HookContext,
    ) -> None:
        elapsed = time.time() - context.metadata.get("_audit_start", 0)
        status = "OK" if result.ok else "FAIL"
        logger.info(
            "Step '%s' %s in %.2fs (rounds=%s)",
            step.id,
            status,
            elapsed,
            result.metrics.get("rounds", "?"),
        )
