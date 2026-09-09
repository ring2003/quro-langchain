"""Configuration loading from environment variables.

Uses python-dotenv to load ``.env`` if present, then falls back to
``os.environ``.  Exposes a frozen ``Settings`` dataclass.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path.cwd() / ".env")


class Settings:
    """Immutable runtime settings sourced from environment variables.

    Environment variables:
        QURO_API_KEY                  (required)
        QURO_BASE_URL                 (default: https://api.openai.com/v1)
        QURO_MODEL                    (default: gpt-4o-mini)
        QURO_TEMPERATURE              (default: 0)
        QURO_MAX_STEPS                (default: 5)
        QURO_MAX_ROUNDS               (default: 12)
        QURO_CONTEXT_BUDGET_CHARS     (default: 12000)
        QURO_REASONING_DISPLAY        (default: post)
        QURO_STATE_DIR                (default: .quro/mvp2)
        QURO_CHECKPOINT_EACH_ROUND    (default: 0)
        QURO_MODEL_PLANNER            (optional)
        QURO_MODEL_WORKER             (optional)
        QURO_MODEL_EVALUATOR          (optional)
        QURO_PROMPT_DIR               (optional — custom Jinja2 template directory)
        QURO_HINT_MAX_CHARS           (default: 160 — per-hint preview max chars)
        QURO_HINTS_MAX_ITEMS          (default: 8 — max hint rows per projection)
        QURO_HINTS_BUDGET_CHARS       (default: 0 = auto, 20% of context budget)
        QURO_PLANNER_MCP_TOOLS        (optional — comma-separated MCP server names for Planner, e.g. "codegraph")
        QURO_WORKER_MCP_TOOLS         (optional — comma-separated MCP server names for Workers, e.g. "codegraph")
        QURO_PLANNER_EXPLORER_TOOLS   (optional — comma-separated explorer tool names for Planner)
        QURO_PLANNER_BUILDER_TOOLS    (optional — comma-separated builder tool names for Planner)
        QURO_RETRY_MAX_ATTEMPTS       (default: None — unlimited API retry attempts)
        QURO_RETRY_BACKOFF_BASE       (default: 1.0 — base backoff seconds for retries)
        QURO_YES                      (default: 0 — skip model-switch confirmation)
        QURO_DEBUG                    (default: 0 — enable debug mode)
        QURO_OVERRIDE                 (default: 0 — override MetaPlanner UNSAT
                                      (simulate continuation); requires QURO_DEBUG=1)
    """

    def __init__(self) -> None:
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._model: str | None = None
        self._temperature: float | None = None
        self._max_steps: int | None = None
        self._max_rounds: int | None = None
        self._context_budget_chars: int | None = None
        self._reasoning_display: str | None = None
        self._state_dir: str | None = None
        self._model_planner: str | None = None
        self._model_worker: str | None = None
        self._model_evaluator: str | None = None
        self._prompt_dir: str | None = None
        self._hint_max_chars: int | None = None
        self._hints_max_items: int | None = None
        self._hints_budget_chars: int | None = None
        self._planner_mcp_tools: list[str] | None = None
        self._worker_mcp_tools: list[str] | None = None
        self._planner_explorer_tools: list[str] | None = None
        self._planner_builder_tools: list[str] | None = None
        self._retry_max_attempts: int | None = None
        self._retry_backoff_base: float | None = None
        self._yes: bool | None = None
        self._debug: bool | None = None
        self._override: bool | None = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = os.getenv("QURO_API_KEY", "")
        return self._api_key

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            self._base_url = os.getenv("QURO_BASE_URL", "https://api.openai.com/v1")
        return self._base_url

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = os.getenv("QURO_MODEL", "gpt-4o-mini")
        return self._model

    @property
    def temperature(self) -> float:
        if self._temperature is None:
            raw = os.getenv("QURO_TEMPERATURE", "0")
            try:
                self._temperature = float(raw)
            except ValueError:
                self._temperature = 0.0
        return self._temperature

    @property
    def max_steps(self) -> int:
        if self._max_steps is None:
            raw = os.getenv("QURO_MAX_STEPS", "5")
            try:
                self._max_steps = int(raw)
            except ValueError:
                self._max_steps = 5
        return self._max_steps

    @property
    def max_rounds(self) -> int:
        if self._max_rounds is None:
            raw = os.getenv("QURO_MAX_ROUNDS", "50")
            try:
                self._max_rounds = int(raw)
            except ValueError:
                self._max_rounds = 50
        return self._max_rounds

    @property
    def context_budget_chars(self) -> int:
        if self._context_budget_chars is None:
            raw = os.getenv("QURO_CONTEXT_BUDGET_CHARS", "12000")
            try:
                self._context_budget_chars = int(raw)
            except ValueError:
                self._context_budget_chars = 12000
        return self._context_budget_chars

    @property
    def reasoning_display(self) -> str:
        if self._reasoning_display is None:
            self._reasoning_display = os.getenv("QURO_REASONING_DISPLAY", "post")
        return self._reasoning_display

    @property
    def state_dir(self) -> str:
        if self._state_dir is None:
            self._state_dir = os.getenv("QURO_STATE_DIR", ".quro/mvp2")
        return self._state_dir

    @property
    def model_planner(self) -> str | None:
        if self._model_planner is None:
            self._model_planner = os.getenv("QURO_MODEL_PLANNER") or None
        return self._model_planner

    @property
    def model_worker(self) -> str | None:
        if self._model_worker is None:
            self._model_worker = os.getenv("QURO_MODEL_WORKER") or None
        return self._model_worker

    @property
    def model_evaluator(self) -> str | None:
        if self._model_evaluator is None:
            self._model_evaluator = os.getenv("QURO_MODEL_EVALUATOR") or None
        return self._model_evaluator

    @property
    def prompt_dir(self) -> str | None:
        if self._prompt_dir is None:
            self._prompt_dir = os.getenv("QURO_PROMPT_DIR") or None
        return self._prompt_dir

    @property
    def hint_max_chars(self) -> int:
        if self._hint_max_chars is None:
            raw = os.getenv("QURO_HINT_MAX_CHARS", "160")
            try:
                self._hint_max_chars = int(raw)
            except ValueError:
                self._hint_max_chars = 160
        return self._hint_max_chars

    @property
    def hints_max_items(self) -> int:
        if self._hints_max_items is None:
            raw = os.getenv("QURO_HINTS_MAX_ITEMS", "8")
            try:
                self._hints_max_items = int(raw)
            except ValueError:
                self._hints_max_items = 8
        return self._hints_max_items

    @property
    def hints_budget_chars(self) -> int:
        if self._hints_budget_chars is None:
            raw = os.getenv("QURO_HINTS_BUDGET_CHARS", "0")
            try:
                self._hints_budget_chars = int(raw)
            except ValueError:
                self._hints_budget_chars = 0
        return self._hints_budget_chars

    @staticmethod
    def _split_csv(raw: str) -> list[str]:
        """Split a comma-separated string into non-empty stripped tokens."""
        if not raw or not raw.strip():
            return []
        return [t.strip() for t in raw.split(",") if t.strip()]

    @property
    def planner_mcp_tools(self) -> list[str]:
        """MCP server names (from .mcp.json) available to the Planner. Empty list means none."""
        if self._planner_mcp_tools is None:
            raw = os.getenv("QURO_PLANNER_MCP_TOOLS", "")
            self._planner_mcp_tools = self._split_csv(raw)
        return self._planner_mcp_tools

    @property
    def worker_mcp_tools(self) -> list[str]:
        """MCP server names (from .mcp.json) available to Workers. Empty list means none."""
        if self._worker_mcp_tools is None:
            raw = os.getenv("QURO_WORKER_MCP_TOOLS", "")
            self._worker_mcp_tools = self._split_csv(raw)
        return self._worker_mcp_tools

    @property
    def planner_explorer_tools(self) -> list[str]:
        """Explorer tool names available to the Planner. Empty list means all (read, ls, grep, find, shell)."""
        if self._planner_explorer_tools is None:
            raw = os.getenv("QURO_PLANNER_EXPLORER_TOOLS", "")
            self._planner_explorer_tools = self._split_csv(raw)
        return self._planner_explorer_tools

    @property
    def planner_builder_tools(self) -> list[str]:
        """Builder tool names available to the Planner. Empty list means none."""
        if self._planner_builder_tools is None:
            raw = os.getenv("QURO_PLANNER_BUILDER_TOOLS", "")
            self._planner_builder_tools = self._split_csv(raw)
        return self._planner_builder_tools

    @property
    def retry_max_attempts(self) -> int | None:
        """Maximum retry attempts for API calls (default: None = unlimited)."""
        if self._retry_max_attempts is None:
            raw = os.getenv("QURO_RETRY_MAX_ATTEMPTS", "")
            if raw:
                try:
                    self._retry_max_attempts = int(raw)
                except ValueError:
                    self._retry_max_attempts = None
            else:
                self._retry_max_attempts = None
        return self._retry_max_attempts

    @property
    def retry_backoff_base(self) -> float:
        """Base backoff seconds for retry attempts (default: 1.0)."""
        if self._retry_backoff_base is None:
            raw = os.getenv("QURO_RETRY_BACKOFF_BASE", "1.0")
            try:
                self._retry_backoff_base = float(raw)
            except ValueError:
                self._retry_backoff_base = 1.0
        return self._retry_backoff_base

    @property
    def yes(self) -> bool:
        """Skip model-switch confirmation (default: False)."""
        if self._yes is None:
            raw = os.getenv("QURO_YES", "0")
            self._yes = raw == "1"
        return self._yes

    @property
    def debug(self) -> bool:
        """Enable debug mode (default: False)."""
        if self._debug is None:
            raw = os.getenv("QURO_DEBUG", "0")
            self._debug = raw == "1"
        return self._debug

    @property
    def override(self) -> bool:
        """Override MetaPlanner SAT/UNSAT decisions (default: False).

        Requires ``debug`` to also be True — the caller must validate
        this constraint.
        """
        if self._override is None:
            raw = os.getenv("QURO_OVERRIDE", "0")
            self._override = raw == "1"
        return self._override
