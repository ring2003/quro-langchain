"""Unit tests for retry module."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from quro.runtime.retry import (
    retry_with_backoff,
    is_retriable,
    _error_description,
    RETRIABLE_EXCEPTIONS,
)

# ---------------------------------------------------------------------------
# Helpers to create APIStatusError mocks
# ---------------------------------------------------------------------------

def _make_api_status_error(status_code: int) -> MagicMock:
    """Create a mock APIStatusError with the given status code."""
    from openai import APIStatusError
    response = MagicMock(status_code=status_code)
    body = {"error": {"message": "test error"}}
    return APIStatusError("test error", response=response, body=body)

# ---------------------------------------------------------------------------
# is_retriable
# ---------------------------------------------------------------------------


def test_is_retriable_429():
    """HTTP 429 is retriable."""
    error = _make_api_status_error(429)
    assert is_retriable(error) is True


def test_is_retriable_500():
    """HTTP 5xx is retriable."""
    error = _make_api_status_error(500)
    assert is_retriable(error) is True


def test_is_retriable_503():
    """HTTP 503 is retriable."""
    error = _make_api_status_error(503)
    assert is_retriable(error) is True


def test_is_retriable_400():
    """HTTP 4xx (except 429) is NOT retriable."""
    error = _make_api_status_error(400)
    assert is_retriable(error) is False


def test_is_retriable_401():
    """HTTP 401 is NOT retriable."""
    error = _make_api_status_error(401)
    assert is_retriable(error) is False

def test_is_retriable_404():
    """HTTP 404 is NOT retriable."""
    error = _make_api_status_error(404)
    assert is_retriable(error) is False


def test_is_retriable_connection_refused():
    """ConnectionRefusedError is retriable."""
    error = ConnectionRefusedError("Connection refused")
    assert is_retriable(error) is True


def test_is_retriable_connection_error():
    """ConnectionError is retriable."""
    error = ConnectionError("Connection error")
    assert is_retriable(error) is True

# ---------------------------------------------------------------------------
# _error_description
# ---------------------------------------------------------------------------


def test_error_description_api_status_error():
    """HTTP status code is included in description."""
    error = _make_api_status_error(503)
    assert "HTTP 503" in _error_description(error)


def test_error_description_connection_refused():
    """ConnectionRefusedError is described correctly."""
    error = ConnectionRefusedError("Connection refused")
    # ConnectionRefusedError is a subclass of ConnectionError, so it returns ConnectionError
    assert _error_description(error) == "ConnectionError"

# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_first_attempt():
    """No retry when the function succeeds immediately."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        return "success"

    result = retry_with_backoff(fn, max_attempts=3, backoff_base=0.001)
    assert result == "success"
    assert call_count[0] == 1


def test_retry_succeeds_after_failures():
    """Retry succeeds after transient failures."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        if call_count[0] < 3:
            raise _make_api_status_error(429)
        return "success"

    result = retry_with_backoff(fn, max_attempts=3, backoff_base=0.001)
    assert result == "success"
    assert call_count[0] == 3


def test_retry_with_unlimited_attempts():
    """Unlimited retries work when max_attempts is None."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        if call_count[0] < 5:
            raise _make_api_status_error(429)
        return "success"

    result = retry_with_backoff(fn, max_attempts=None, backoff_base=0.001)
    assert result == "success"
    assert call_count[0] == 5


def test_retry_connection_refused_is_retriable():
    """ConnectionRefusedError triggers retry."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionRefusedError("Connection refused")
        return "success"

    result = retry_with_backoff(fn, max_attempts=3, backoff_base=0.001)
    assert result == "success"
    assert call_count[0] == 2


def test_retry_non_retriable_error_propagates_immediately():
    """Non-retriable errors propagate immediately without retry."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        raise _make_api_status_error(400)

    with pytest.raises(BaseException):
        retry_with_backoff(fn, max_attempts=3, backoff_base=0.001)
    assert call_count[0] == 1
