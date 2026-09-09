"""Runtime backend abstraction.

IRuntimeBackend is a Protocol that any LLM backend must implement.
Domain layer depends only on this — not on any specific SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class CompletionResult:
    content: str
    reasoning: str = ""
    raw_content: str = ""
    usage: TokenUsage | None = None
    tool_calls: list[dict] = field(default_factory=list)


@runtime_checkable
class IRuntimeBackend(Protocol):
    """Interface for LLM completion backends.

    MVP only requires single-turn completion.
    Future: stream(), complete_messages(), etc.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> str:
        """Single-turn completion (legacy: returns content only).

        Args:
            prompt: The user/instruction prompt.
            system: Optional system message.
            tools: Optional list of tools to bind for tool calling.
                   If provided, the backend should run a ReAct loop
                   (model → tool → model → ...) and return the final text.

        Returns:
            Assistant's response text.
        """
        ...

    def complete_with_meta(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        """MVP-2 preferred: content + reasoning + usage.

        Args:
            log_prefix: A prefix like ``[planner/1]> `` prepended to all display output.
        """
        ...

    def complete_one(
        self,
        messages: list[dict],
        *,
        tools: Optional[list] = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        """Single completion turn: model -> response (possibly with tool_calls).

        Unlike complete_with_meta(), this does NOT run a ReAct loop.
        The caller is responsible for feeding tool results back.

        Args:
            messages: Full message list including system, user, assistant, tool messages.
            tools: Optional list of tools to bind for tool calling.
            log_prefix: A prefix like ``[planner/1]> `` prepended to all display output.

        Returns:
            CompletionResult with content and/or tool_calls.
        """
        ...


class FakeBackend:
    """A fake backend that returns canned responses in sequence.

    Parameters:
        sequence: List of entries. Each entry is either:
            - A string: plain text response (no tool calls).
            - A list of dicts: tool call sequence where each dict has
              ``name`` and ``args`` keys.

        responses: (legacy) list of plain text responses.
        response_meta: (legacy) list of CompletionResult objects.
        tool_call_sequence: (legacy) list of list[dict] for tool calls.

    Useful for testing without network calls.
    """

    def __init__(
        self,
        sequence: list | None = None,
        responses: list[str] | None = None,
        response_meta: list[CompletionResult] | None = None,
        tool_call_sequence: list[list[dict[str, Any]]] | None = None,
    ):
        if sequence is not None:
            self._sequence: list = list(sequence)
        else:
            self._sequence = []
        self.responses = list(responses) if responses else []
        self.response_meta = list(response_meta) if response_meta else []
        self.tool_call_sequence = list(tool_call_sequence) if tool_call_sequence else []
        self.call_count = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list | None = None,
    ) -> str:
        return self.complete_with_meta(prompt, system=system, tools=tools).content

    def complete_with_meta(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list | None = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        if self._sequence:
            idx = self.call_count % len(self._sequence)
            self.call_count += 1
            entry = self._sequence[idx]

            if isinstance(entry, str):
                return CompletionResult(content=entry)

            if isinstance(entry, list) and tools:
                tool_map = {t.name: t for t in tools}
                for tc in entry:
                    tool = tool_map.get(tc["name"])
                    if tool is not None:
                        try:
                            tool.invoke(tc["args"])
                        except Exception:
                            pass
                return CompletionResult(content="ok")

            return CompletionResult(content=str(entry))

        if self.response_meta:
            idx = self.call_count % len(self.response_meta)
            self.call_count += 1
            return self.response_meta[idx]

        if tools and self.tool_call_sequence:
            idx = self.call_count % len(self.tool_call_sequence)
            self.call_count += 1
            calls = self.tool_call_sequence[idx]
            tool_map = {t.name: t for t in tools}
            for tc in calls:
                tool = tool_map.get(tc["name"])
                if tool is not None:
                    try:
                        tool.invoke(tc["args"])
                    except Exception:
                        pass
            return CompletionResult(content="ok")

        self.call_count += 1
        if not self.responses:
            return CompletionResult(content="ok")
        idx = (self.call_count - 1) % len(self.responses)
        return CompletionResult(content=self.responses[idx])

    def complete_one(
        self,
        messages: list[dict],
        *,
        tools: list | None = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        if self._sequence:
            idx = self.call_count % len(self._sequence)
            self.call_count += 1
            entry = self._sequence[idx]

            if isinstance(entry, str):
                return CompletionResult(content=entry)

            if isinstance(entry, list):
                return CompletionResult(content="ok", tool_calls=list(entry))

            return CompletionResult(content=str(entry))

        self.call_count += 1
        return CompletionResult(content="ok")

    def add_response(self, text: str) -> None:
        """Append a canned response."""
        self.responses.append(text)

    def add_meta_response(self, cr: CompletionResult) -> None:
        self.response_meta.append(cr)
