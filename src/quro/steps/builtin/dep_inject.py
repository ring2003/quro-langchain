"""Dependency inject hook — rewrite step objective with prior step context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from quro.steps.hooks import HookContext

if TYPE_CHECKING:
    from quro.steps.core import StepResult, StepSpec


@dataclass
class DependencyInjectHook:
    """Inject prior step artifacts into the step's objective.

    Usage in YAML::

        pre_hooks:
          - name: dependency_inject
            config:
              max_artifact_chars: 500

    Args:
        max_artifact_chars: Maximum characters of each artifact body to inject.
    """

    name: str = "dependency_inject"
    max_artifact_chars: int = 500

    def on_pre_step(
        self, step: StepSpec, context: HookContext
    ) -> StepSpec | None:
        fragments: list[str] = []
        for dep_id in step.depends_on:
            dep_result = context.step_results.get(dep_id)
            if dep_result is None:
                continue
            for artifact in getattr(dep_result, "artifacts", []) or []:
                summary = artifact.get("summary", "")
                body = artifact.get("body", "")
                line = f"[{dep_id}] {summary}" if summary else f"[{dep_id}]"
                if body:
                    line = f"{line}\n    {body[:self.max_artifact_chars]}"
                fragments.append(line)

        if not fragments:
            return None

        injected = (
            f"{step.objective}\n\n"
            f"CONTEXT FROM PRIOR STEPS:\n" + "\n".join(fragments)
        )

        from dataclasses import replace

        return replace(step, objective=injected)
