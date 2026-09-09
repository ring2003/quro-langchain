"""Unit tests for confirm module."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from quro.core.confirm import confirm_model_switch

# ---------------------------------------------------------------------------
# confirm_model_switch
# ---------------------------------------------------------------------------


def test_confirm_auto_mode():
    """In auto mode, confirmation is skipped and True is returned."""
    result = confirm_model_switch(
        task_id="planner/understanding/s-001",
        from_model="gpt-4o-mini",
        to_model="gpt-4o",
        auto_confirm=True,
    )
    assert result is True
