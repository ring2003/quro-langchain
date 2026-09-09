"""Tests for logical resource descriptors (refs.py)."""

from __future__ import annotations

import pytest

from quro.core.resources import InvalidDescriptor, ResourceRef, parse_descriptor


def test_parse_simple():
    assert parse_descriptor("artifact://art_x") == ResourceRef("artifact", "art_x")


def test_parse_with_subpath():
    assert parse_descriptor("clue://s1/clues") == ResourceRef(
        "clue", "s1", ("clues",)
    )


def test_parse_roundtrip():
    for s in [
        "step://s1",
        "artifact://art_ab12cd34",
        "clue://s1/clues",
        "round://3/step/s1",
        "plan://main",
        "interpretation://main",
    ]:
        assert str(parse_descriptor(s)) == s


def test_parse_round_with_subpath():
    assert parse_descriptor("round://3/step/s1") == ResourceRef(
        "round", "3", ("step", "s1")
    )


def test_invalid_missing_separator():
    with pytest.raises(InvalidDescriptor):
        parse_descriptor("bad")


def test_invalid_unknown_kind():
    with pytest.raises(InvalidDescriptor):
        parse_descriptor("file://x")


def test_invalid_empty_id():
    with pytest.raises(InvalidDescriptor):
        parse_descriptor("artifact://")


def test_invalid_non_string():
    with pytest.raises(InvalidDescriptor):
        parse_descriptor(123)
