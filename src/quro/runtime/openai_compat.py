"""OpenAI-compatible runtime backend with reasoning_content support.

Uses the raw ``openai.OpenAI`` client instead of LangChain ``ChatOpenAI``.

Why not ChatOpenAI?
    ``langchain-openai`` deliberately targets the official OpenAI schema only.
    Non-standard third-party fields such as ``reasoning_content`` /
    ``reasoning_details`` (DeepSeek, vLLM, MLX, OpenRouter, …) are dropped in
    both ``_convert_dict_to_message`` and ``_convert_delta_to_message_chunk``.
    That made stream/post display of model thinking always empty even when the
    server streamed reasoning correctly.

This backend:
  1. Streams chat completions via the OpenAI SDK (``extra='allow'`` preserves
     provider fields on deltas/messages).
  2. Accumulates ``reasoning_content`` (and falls back to ``<think>`` tags).
  3. Honours display modes: ``off`` | ``post`` | ``stream``.
  4. Runs a ReAct tool loop when tools are provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from quro.runtime.base import CompletionResult, TokenUsage
from quro.runtime.display import format_reasoning
from quro.runtime.retry import retry_with_backoff, is_retriable, _error_description


MAX_TOOL_RESULT_CHARS = 8192  # rough approximation of 2048 tokens


def _is_dump_context_enabled() -> bool:
    """Check if context dumping is enabled (QURO_DEBUG or QURO_DUMP_CONTEXT)."""
    import os
    if os.environ.get("QURO_DEBUG") in ("1", "true", "True", "yes"):
        return True
    return os.environ.get("QURO_DUMP_CONTEXT") in ("1", "true", "True", "yes")


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without network)
# ---------------------------------------------------------------------------


def _extract_reasoning(content: str) -> tuple[str, str]:
    """Split ``<think>...</think>`` from content. Returns (reasoning, clean)."""
    text = content or ""
    stripped = text.replace("<think/>", "").replace("<think />", "")
    if stripped != text:
        return "", stripped.strip()
    if "</think>" in text:
        before = text.split("</think>", 1)[0]
        after = text.split("</think>", 1)[1]
        reasoning = ""
        if "<think>" in before:
            reasoning = before.split("<think>", 1)[1].lstrip("\n")
        return reasoning.rstrip("\n"), after.lstrip("\n")
    return "", text


def _get_field(obj: Any, *names: str) -> str:
    """Read a possibly non-standard field from an OpenAI SDK model / dict."""
    if obj is None:
        return ""
    data: dict[str, Any] = {}
    if isinstance(obj, dict):
        data = obj
    else:
        if hasattr(obj, "model_dump"):
            try:
                data = obj.model_dump(exclude_unset=False)
            except Exception:
                data = {}
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                data.setdefault(k, v)
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val:
                    return str(val)
    for name in names:
        val = data.get(name)
        if val:
            return str(val)
    return ""


def _tools_to_openai(tools: list) -> list[dict[str, Any]]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [convert_to_openai_tool(t) for t in tools]


@dataclass
class _StreamAccumulator:
    """Mutable accumulator for one streamed chat completion."""

    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    # index -> {id, name, arguments}
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    usage: TokenUsage | None = None
    # For live <think> parsing of content deltas when no reasoning_content field
    _in_think: bool = False
    _think_buf: list[str] = field(default_factory=list)
    _content_buf: list[str] = field(default_factory=list)
    # stream display state
    _header_printed: bool = False
    _streamed_reasoning_field: bool = False
    log_prefix: str = ""

    def feed_delta(
        self,
        delta: Any,
        *,
        display_mode: str = "post",
    ) -> None:
        rc = _get_field(delta, "reasoning_content", "reasoning")
        if rc:
            self.reasoning_parts.append(rc)
            self._streamed_reasoning_field = True
            if display_mode == "stream":
                self._print_reasoning_token(rc)

        content = None
        if isinstance(delta, dict):
            content = delta.get("content")
        else:
            content = getattr(delta, "content", None)
        if content:
            self.content_parts.append(content)
            # Live <think> handling only when provider does not send reasoning_content
            if not self._streamed_reasoning_field:
                self._feed_content_for_think(content, display_mode=display_mode)

        raw_tcs = None
        if isinstance(delta, dict):
            raw_tcs = delta.get("tool_calls")
        else:
            raw_tcs = getattr(delta, "tool_calls", None)
        if raw_tcs:
            self._feed_tool_calls(raw_tcs)

    def feed_usage(self, usage: Any) -> None:
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        if not reasoning_tokens and isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", prompt) or 0)
            completion = int(usage.get("completion_tokens", completion) or 0)
            ctd = usage.get("completion_tokens_details") or {}
            if isinstance(ctd, dict):
                reasoning_tokens = int(ctd.get("reasoning_tokens", 0) or 0)
        self.usage = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning_tokens,
        )

    def _print_reasoning_token(self, token: str) -> None:
        if not self._header_printed:
            print(f"{self.log_prefix}reasoning:", end="", flush=True)
            self._header_printed = True
        print(token, end="", flush=True)

    def _feed_content_for_think(self, token: str, *, display_mode: str) -> None:
        """Parse content deltas for ``<think>`` tags; optionally stream the body."""
        i = 0
        while i < len(token):
            if self._in_think:
                end = token.find("</think>", i)
                if end == -1:
                    piece = token[i:]
                    self._think_buf.append(piece)
                    if display_mode == "stream":
                        self._print_reasoning_token(piece)
                    i = len(token)
                else:
                    piece = token[i:end]
                    if piece:
                        self._think_buf.append(piece)
                        if display_mode == "stream":
                            self._print_reasoning_token(piece)
                    self._in_think = False
                    i = end + len("</think>")
            else:
                start = token.find("<think", i)
                if start == -1:
                    self._content_buf.append(token[i:])
                    i = len(token)
                else:
                    if start > i:
                        self._content_buf.append(token[i:start])
                    # Self-closing or open tag
                    rest = token[start:]
                    if rest.startswith("<think/>") or rest.startswith("<think />"):
                        # skip empty think
                        close = rest.find(">")
                        i = start + close + 1
                    elif rest.startswith("<think>"):
                        self._in_think = True
                        i = start + len("<think>")
                    else:
                        # partial tag at end of chunk — keep as content for now
                        self._content_buf.append(token[i:])
                        i = len(token)

    def _feed_tool_calls(self, raw_tcs: list) -> None:
        for tc in raw_tcs:
            if isinstance(tc, dict):
                idx = int(tc.get("index", 0) or 0)
                tc_id = tc.get("id") or ""
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                arguments = fn.get("arguments") or ""
            else:
                idx = int(getattr(tc, "index", 0) or 0)
                tc_id = getattr(tc, "id", None) or ""
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) or "" if fn else ""
                arguments = getattr(fn, "arguments", None) or "" if fn else ""
            slot = self.tool_calls.setdefault(
                idx, {"id": "", "name": "", "arguments": ""}
            )
            if tc_id:
                slot["id"] = tc_id
            if name:
                slot["name"] = name
            if arguments:
                slot["arguments"] += arguments

    def finish_stream_line(self, display_mode: str) -> None:
        if display_mode == "stream" and self._header_printed:
            print(flush=True)

    def to_result(self) -> tuple[str, str, list[dict[str, Any]], TokenUsage | None]:
        raw_content = "".join(self.content_parts)
        reasoning = "".join(self.reasoning_parts)

        if reasoning:
            content = raw_content
        elif self._think_buf or self._content_buf:
            # Prefer the live-parsed split when we walked content for <think>
            reasoning = "".join(self._think_buf).rstrip("\n")
            content = "".join(self._content_buf)
            if not reasoning:
                reasoning, content = _extract_reasoning(raw_content)
        else:
            reasoning, content = _extract_reasoning(raw_content)

        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(self.tool_calls):
            acc = self.tool_calls[idx]
            args_raw = acc["arguments"] or "{}"
            try:
                args: Any = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(
                {
                    "id": acc["id"] or f"call_{idx}",
                    "name": acc["name"],
                    "args": args if isinstance(args, dict) else {"value": args},
                }
            )
        return content, reasoning, tool_calls, self.usage


def _assistant_api_message(
    content: str,
    reasoning: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an OpenAI-format assistant message for multi-turn tool loops."""
    msg: dict[str, Any] = {"role": "assistant"}
    if tool_calls and not content:
        msg["content"] = None
    else:
        msg["content"] = content
    # Re-send reasoning_content when the provider emitted it — some thinking
    # backends (DeepSeek-R1 style) expect it in subsequent turns.
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False)
                    if isinstance(tc["args"], (dict, list))
                    else str(tc["args"]),
                },
            }
            for tc in tool_calls
        ]
    return msg


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OpenAICompatibleBackend:
    """LLM backend over any OpenAI-compatible Chat Completions endpoint.

    Preserves provider ``reasoning_content`` that LangChain ``ChatOpenAI`` drops.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        display_mode: str = "post",
        max_tokens: int = 4096,
        max_tool_rounds: int = 10,
        client: OpenAI | None = None,
        *,
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 1.0,
    ) -> None:
        self.model_name = model
        self._temperature = temperature
        self._display_mode = display_mode
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base = retry_backoff_base
        self._client = client or OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    def complete_one(
        self,
        messages: list[dict],
        *,
        tools: list | None = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        #for m in messages:
        #    if m.get("role") == "system":
        #        print(f"{log_prefix}[SYSTEM] {m['content'][:200]}")
        openai_tools = _tools_to_openai(tools) if tools else None
        content, reasoning, tool_calls, usage, raw_content = self._stream_once(
            messages, tools=openai_tools, log_prefix=log_prefix,
        )

        if self._display_mode == "post" and reasoning:
            print(f"{log_prefix}{format_reasoning(reasoning).lstrip()}")

        if tool_calls:
            for tc in tool_calls:
                args_preview = json.dumps(tc["args"], ensure_ascii=False)
                if len(args_preview) > 120:
                    args_preview = args_preview[:120] + "..."
                print(f"{log_prefix}[Tool call] {tc['name']}({args_preview})")

        return CompletionResult(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            raw_content=raw_content,
            usage=usage,
        )

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> str:
        return self.complete_with_meta(prompt, system=system, tools=tools).content

    def complete_with_meta(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
        log_prefix: str = "",
    ) -> CompletionResult:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        openai_tools = _tools_to_openai(tools) if tools else None
        tool_map = {t.name: t for t in tools} if tools else {}

        last_content = ""
        last_reasoning = ""
        last_raw = ""
        last_usage: TokenUsage | None = None

        rounds = self._max_tool_rounds if openai_tools else 1
        for _ in range(rounds):
            content, reasoning, tool_calls, usage, raw_content = self._stream_once(
                messages, tools=openai_tools, log_prefix=log_prefix,
            )
            last_content = content
            last_reasoning = reasoning
            last_raw = raw_content
            if usage is not None:
                last_usage = usage

            if self._display_mode == "post" and reasoning:
                print(f"{log_prefix}{format_reasoning(reasoning).lstrip()}")

            if not tool_calls:
                return CompletionResult(
                    content=content,
                    reasoning=reasoning,
                    raw_content=raw_content,
                    usage=usage,
                )

            messages.append(_assistant_api_message(content, reasoning, tool_calls))
            for tc in tool_calls:
                args_preview = json.dumps(tc["args"], ensure_ascii=False)
                if len(args_preview) > 120:
                    args_preview = args_preview[:120] + "..."
                print(f"{log_prefix}[Tool call] {tc['name']}({args_preview})")

                tool = tool_map.get(tc["name"])
                if tool is None:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        result = tool.invoke(tc["args"])
                    except Exception as e:
                        result = f"Error executing {tc['name']}: {e}"

                result_str = str(result)
                if _is_dump_context_enabled():
                    _preview = result_str[:500]
                    if len(result_str) > 500:
                        _preview += f"... ({len(result_str)} chars total)"
                    print(f"{log_prefix}[Tool result] {_preview}")
                if len(result_str) > MAX_TOOL_RESULT_CHARS:
                    preview = result_str[:200]
                    result = (
                        f"Error: tool result exceeds {MAX_TOOL_RESULT_CHARS} chars "
                        f"(~2048 tokens). Got {len(result_str)} chars. "
                        f"Use a more specific command. Preview:\n{preview}"
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )

        return CompletionResult(
            content=last_content,
            reasoning=last_reasoning,
            raw_content=last_raw,
            usage=last_usage,
        )

    def _stream_once(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        log_prefix: str = "",
    ) -> tuple[str, str, list[dict[str, Any]], TokenUsage | None, str]:
        """One streamed completion. Returns content, reasoning, tool_calls, usage, raw.

        Wrapped with retry_with_backoff to handle transient API errors.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        def _raw_create() -> tuple[str, str, list[dict[str, Any]], TokenUsage | None, str]:
            """Execute the API call and accumulate the stream."""
            acc = _StreamAccumulator(log_prefix=log_prefix)
            raw_content_parts: list[str] = []

            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    acc.feed_usage(chunk.usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                # Track raw content independently for raw_content field
                c = getattr(delta, "content", None) if delta is not None else None
                if c:
                    raw_content_parts.append(c)
                if delta is not None:
                    acc.feed_delta(delta, display_mode=self._display_mode)

            acc.finish_stream_line(self._display_mode)
            content, reasoning, tool_calls, usage = acc.to_result()
            raw_content = "".join(raw_content_parts)
            return content, reasoning, tool_calls, usage, raw_content

        def _on_retry(attempt: int, error: BaseException, delay: float) -> None:
            """Callback for retry logging."""
            error_desc = _error_description(error)
            print(f"{log_prefix}⚠️ API retry {attempt}/{self._retry_max_attempts} "
                  f"({error_desc}, backoff {delay:.1f}s)")

        return retry_with_backoff(
            _raw_create,
            max_attempts=self._retry_max_attempts,
            backoff_base=self._retry_backoff_base,
            jitter=True,
            retriable=is_retriable,
            on_retry=_on_retry,
        )
