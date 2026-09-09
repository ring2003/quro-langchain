from __future__ import annotations

from enum import Enum

from quro.runtime.base import CompletionResult


class DisplayMode(str, Enum):
    OFF = "off"
    POST = "post"
    STREAM = "stream"


SEPARATOR = "\n" + "\u2500" * 50 + "\n"


def format_reasoning(reasoning: str) -> str:
    lines = reasoning.rstrip().splitlines()
    if not lines:
        return ""
    return SEPARATOR + "  Reasoning:" + SEPARATOR + "\n".join(
        "  " + l for l in lines
    ) + SEPARATOR


def display_reasoning(result: CompletionResult, mode: DisplayMode = DisplayMode.POST) -> None:
    if mode == DisplayMode.OFF or not result.reasoning:
        return
    if mode == DisplayMode.POST:
        print(format_reasoning(result.reasoning))
