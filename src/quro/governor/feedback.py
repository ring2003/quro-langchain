"""FailureFeedback — injects kernel failure memory into prompts.

Fixes R§8: makes ``failed_attempts_detail`` and ``axiom_store`` visible
to the model so it does not repeat already-failed calls (DEP-gate loops,
re-exploration).

See ``docs/system-analysis/09-refactoring-implementation-plan.md`` §4.5.
"""

from __future__ import annotations

from typing import Any


class FailureFeedback:
    """Build a digest of prior failures and axioms for prompt injection.

    Surfaces:
    1. Kernel structured failures (``session.failed_attempts_detail``).
    2. Axioms surviving backtrack (``session.axiom_store``).
    3. Governor-local record of rejected tool calls this step.
    """

    def __init__(
        self,
        *,
        max_attempts_shown: int = 5,
        budget_chars: int = 3000,
    ) -> None:
        self.max_attempts_shown = max_attempts_shown
        self.budget_chars = budget_chars
        self._recent_rejections: list[tuple[str, str]] = []

    def record_rejection(self, tool_name: str, error: str) -> None:
        """Record a rejected tool call for later injection."""
        self._recent_rejections.append((tool_name, error))
        # Keep bounded
        if len(self._recent_rejections) > self.max_attempts_shown * 2:
            self._recent_rejections = self._recent_rejections[-self.max_attempts_shown:]

    def digest(self, session: Any) -> str:
        """Build a feedback digest string for prompt injection."""
        parts: list[str] = []

        # 1. Kernel structured failures
        failed = getattr(session, "failed_attempts_detail", {})
        if failed:
            items = list(failed.items())[-self.max_attempts_shown:]
            for key, reason in items:
                parts.append(f"- tried {key} → {reason}")

        # 2. Axioms from axiom_store
        axioms = getattr(session, "axiom_store", [])
        if axioms:
            for ax in axioms[-self.max_attempts_shown:]:
                if isinstance(ax, dict):
                    parts.append(f"- AXIOM: {ax.get('fact', ax)}")
                else:
                    parts.append(f"- AXIOM: {ax}")

        # 3. Governor-local recent rejections
        for name, err in self._recent_rejections[-self.max_attempts_shown:]:
            parts.append(
                f"- tool '{name}' was rejected: {err} — do NOT retry the same args"
            )

        if not parts:
            return ""

        header = "PREVIOUS ATTEMPTS (do not repeat):\n"
        body = "\n".join(parts)
        full = header + body

        # Budget cap
        if len(full) > self.budget_chars:
            full = full[:self.budget_chars] + "\n... (truncated)"

        return full
