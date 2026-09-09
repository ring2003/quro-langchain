"""Human-in-loop step — a step that pauses the pipeline for human input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quro.steps.core import StepSpec


@runtime_checkable
class HILBackend(Protocol):
    """Backend for human-in-the-loop interaction.

    Decouples the pipeline from any specific UI (CLI, WebSocket, Slack, etc.).
    """

    def ask_user(
        self,
        question: str,
        *,
        choices: list[str] | None = None,
        timeout: int = 0,
    ) -> str:
        """Present *question* to the user and return their response.

        Args:
            question: The question to display.
            choices: Optional list of valid responses.
            timeout: Max seconds to wait (0 = indefinite).

        Returns:
            The user's response as a string.
        """
        ...


@dataclass
class HumanInLoopStep(StepSpec):
    """A step that pauses the pipeline and waits for human input.

    Unlike a regular ``StepSpec``, no LLM backend is invoked — the pipeline
    runner presents the question and collects a response, which is stored as
    an artifact for downstream steps.

    Usage in YAML::

        steps:
          - id: plan_approval
            kind: human_in_loop
            objective: "Please review the generated plan."
            choices: [approve, reject, modify]
            timeout: 300

    Args:
        prompt_template: Jinja2 template for rendering the prompt. Has access
            to ``objective`` and ``step_results``.
        choices: Optional constrained choices for the user.
        timeout: Max seconds to wait for input (0 = indefinite).

    Human-in-the-loop identity is the explicit ``hil`` flag on ``StepSpec``
    (replaces the former ``role == "human"``); this class is the HIL render
    surface only.
    """

    prompt_template: str = ""
    choices: list[str] = field(default_factory=list)
    timeout: int = 0

    def render_prompt(self, context: dict[str, Any]) -> str:
        """Render the prompt template with pipeline context.

        Args:
            context: The pipeline context dict (``problem``, ``results``).

        Returns:
            The rendered prompt string.
        """
        if self.prompt_template:
            try:
                from jinja2 import Template

                tmpl = Template(self.prompt_template)
                return tmpl.render(
                    objective=self.objective,
                    context=context,
                    step_results=context.get("results", {}),
                )
            except ImportError:
                pass
        return self.objective
