"""Human-in-loop tool — ``ask_user`` tool for LLM-driven user queries.

Unlike ``HumanInLoopStep`` (which is a pipeline-level construct that pauses
the entire pipeline), ``ask_user`` is a tool that the LLM can call during its
ReAct loop *within* a step.  The tool call blocks on the HIL backend, and the
response is injected as a tool message so the LLM can continue reasoning.
"""

from __future__ import annotations

from typing import Any

from quro.steps.hil_step import HILBackend


def make_hil_tool(backend: HILBackend) -> Any:
    """Create the ``ask_user`` tool for human-in-the-loop interaction.

    This tool is added to every step's tool set so the LLM can request
    human input at any point during reasoning.

    Args:
        backend: The HIL backend that handles user interaction.

    Returns:
        A ``StructuredTool`` that the LLM can call as ``ask_user``.
    """
    from langchain_core.tools import StructuredTool

    def ask_user(question: str, choices: str = "") -> str:
        """Ask the human user a question during step execution.

        Use this when you need clarification, preference, or approval
        before proceeding.  The pipeline will pause until the user responds.

        Args:
            question: The question to ask the user. Be specific.
            choices: Optional comma-separated list of valid responses.
                     Example: "yes, no, maybe"

        Returns:
            The user's response as a string.
        """
        choice_list = (
            [c.strip() for c in choices.split(",") if c.strip()]
            if choices
            else None
        )
        response = backend.ask_user(question, choices=choice_list)
        return response

    return StructuredTool.from_function(
        func=ask_user,
        name="ask_user",
        description=(
            "Ask the human user a question and wait for their response. "
            "Use this when you need clarification, preference, or approval "
            "before proceeding."
        ),
    )
