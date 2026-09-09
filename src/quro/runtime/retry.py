"""OpenAI API retry with exponential backoff.

Wraps transient API errors (429 Rate Limit, 5xx Server Error, timeouts,
connection refused) with configurable retry attempts and exponential backoff.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from openai import APIError, APITimeoutError, APIStatusError

try:
    from httpx import RemoteProtocolError as _HttpxRemoteProtocolError
except ImportError:  # pragma: no cover
    _HttpxRemoteProtocolError = None  # type: ignore[assignment,misc]

T = TypeVar("T")

RETRIABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    ConnectionRefusedError,
    *(() if _HttpxRemoteProtocolError is None else (_HttpxRemoteProtocolError,)),
)

# Map of status codes → error type names for logging
_STATUS_TYPE_MAP: dict[int, str] = {
    429: "RateLimitError",
}

# Additional OpenAI SDK error types that are retriable
_OPENAI_RETRIABLE: tuple[type[BaseException], ...] = (
    APIError,
    APITimeoutError,
)

# Retryable status codes
_RETRIABLE_STATUS_CODES: set[int] = {429} | {s for s in range(500, 600)}  # 5xx + 429


def is_retriable(error: BaseException) -> bool:
    """Determine whether *error* is a transient error suitable for retry.

    4xx errors (except 429) are NOT retriable — they are client errors.
    """
    # Check OpenAI SDK error types
    if isinstance(error, _OPENAI_RETRIABLE):
        # APIError covers both 4xx and 5xx — check status code
        if isinstance(error, APIStatusError):
            return error.status_code in _RETRIABLE_STATUS_CODES
        return True

    # Check for standard HTTP errors
    if isinstance(error, APIStatusError):
        return error.status_code in _RETRIABLE_STATUS_CODES

    # Standard networking errors
    if isinstance(error, (ConnectionError, TimeoutError, OSError, ConnectionRefusedError)):
        return True

    # httpx transport errors (e.g. incomplete chunked read during SSE streaming)
    if _HttpxRemoteProtocolError is not None and isinstance(error, _HttpxRemoteProtocolError):
        return True

    return False


def _error_description(error: BaseException) -> str:
    """Return a human-readable description of the error for logging."""
    if isinstance(error, APIStatusError):
        status = getattr(error, "status_code", "?")
        return f"HTTP {status}"
    if isinstance(error, APITimeoutError):
        return "TimeoutError"
    if isinstance(error, APIError):
        return "APIError"
    if isinstance(error, ConnectionError):
        return "ConnectionError"
    if isinstance(error, TimeoutError):
        return "TimeoutError"
    if isinstance(error, OSError):
        return f"OSError({error})"
    if _HttpxRemoteProtocolError is not None and isinstance(error, _HttpxRemoteProtocolError):
        return "RemoteProtocolError"
    return type(error).__name__


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int | None = None,
    backoff_base: float = 1.0,
    jitter: bool = True,
    retriable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Retry *fn* on transient API errors with exponential backoff.

    Args:
        fn: The function to execute. It will be called until success or non-retriable error.
        max_attempts: Maximum number of attempts (default: None = unlimited).
        backoff_base: Base delay in seconds for exponential backoff (default: 1.0).
            Actual delay = backoff_base * 2^(attempt-1) + optional jitter.
        jitter: Whether to add random jitter to the backoff delay (default: True).
        retriable: Optional custom predicate to determine if an error is retriable.
            If None, uses the default :func:`is_retriable`.
        on_retry: Optional callback called before each retry.
            Signature: on_retry(attempt, error, delay) -> None

    Returns:
        The return value of *fn* on success.

    Raises:
        The original exception after *max_attempts* failures, or if *max_attempts* is
        None, the original non-retriable exception.
    """
    retriable_fn = retriable or is_retriable

    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as e:
            if max_attempts is not None and attempt >= max_attempts:
                raise  # Max attempts reached — propagate

            if not retriable_fn(e):
                raise  # Non-retriable error — propagate immediately

            # Calculate backoff delay
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter:
                jitter_amount = random.uniform(0, delay * 0.1)
                delay += jitter_amount

            # Call retry callback if provided
            if on_retry:
                on_retry(attempt, e, delay)

            # Sleep before retry
            time.sleep(delay)
