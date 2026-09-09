from __future__ import annotations

import pytest

from quro.runtime.base import CompletionResult, TokenUsage, FakeBackend


class TestCompletionResult:
    def test_content_only(self):
        cr = CompletionResult(content="hello")
        assert cr.content == "hello"
        assert cr.reasoning == ""
        assert cr.raw_content == ""

    def test_with_reasoning(self):
        cr = CompletionResult(content="answer", reasoning="thinking...")
        assert cr.content == "answer"
        assert cr.reasoning == "thinking..."

    def test_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=5)
        cr = CompletionResult(content="x", reasoning="r", usage=usage)
        assert cr.usage is not None
        assert cr.usage.prompt_tokens == 10
        assert cr.usage.completion_tokens == 20
        assert cr.usage.reasoning_tokens == 5


class TestFakeBackendMeta:
    def test_complete_with_meta_fallback(self):
        fb = FakeBackend(responses=["hello world"])
        result = fb.complete_with_meta("prompt")
        assert result.content == "hello world"
        assert result.reasoning == ""

    def test_complete_with_meta_explicit(self):
        fb = FakeBackend(response_meta=[
            CompletionResult(content="ans", reasoning="reason", usage=TokenUsage(prompt_tokens=5)),
        ])
        result = fb.complete_with_meta("prompt")
        assert result.content == "ans"
        assert result.reasoning == "reason"
        assert result.usage and result.usage.prompt_tokens == 5

    def test_legacy_complete_still_works(self):
        fb = FakeBackend(responses=["legacy"])
        assert fb.complete("prompt") == "legacy"
        assert fb.call_count == 1

    def test_reasoning_absent_in_content(self):
        fb = FakeBackend(response_meta=[
            CompletionResult(content="clean output", reasoning="hidden thought"),
        ])
        result = fb.complete_with_meta("prompt")
        assert result.content == "clean output"
        assert result.reasoning == "hidden thought"
        assert "hidden" not in result.content

    def test_cycle_through_responses(self):
        fb = FakeBackend(responses=["a", "b"])
        assert fb.complete_with_meta("p").content == "a"
        assert fb.complete_with_meta("p").content == "b"
        assert fb.complete_with_meta("p").content == "a"
