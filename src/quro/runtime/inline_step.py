"""InlineStep — a one-shot, kernel-free reasoning unit (primitive-step.md).

An inline step is a lightweight built-in derived from a ``StepType`` that the
recovery path drives **directly**: it carries its own prompt, runs as a single
LLM call with no ``ReasoningSession``, no pipeline run, and no domain write
tools, and its output (a clean next-action text) is fed back as the enclosing
step's input.

Two hard rules keep the tier stable (primitive-step.md §5):

1. **Stateless** — the inline step persists nothing; its one-shot call lives on
   the stack and its output rides the prompt (decision B).  A re-interrupt
   simply re-evaluates it.
2. **No domain write tools** — enforced by construction: the inline step's tool
   surface is empty (read-only projection only, and even that is unused by the
   built-ins).

InlineStep has two implementations (primitive-step.md §3):

- **Compaction** (non-invasive recovery) — the objective is frozen; the act
  only injects a "how to continue" instruction.  The R1 recovery path uses
  this; built-ins ``backtrack_plan`` / ``next_step`` live here.
- **Steering** (invasive exploration loop) — the objective is progressively
  overridden; see ``steering.py``.

Inline-step types are framework built-ins registered by name
(primitive-step.md §6); a domain declares *which* compaction to inline on
resume via ``recovery_compaction`` / ``recovery_compaction_for_step_type``.
The runtime resolves ``recovery_compaction → InlineStepRegistry → eval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InlineStep:
    """A one-shot, kernel-free reasoning unit derived from a ``StepType``.

    Attributes:
        name: Built-in inline-step name (e.g. ``"backtrack_plan"``).
        step_type: The ``StepType`` it borrows identity from (for ACL / audit /
            ``OWNS``).  Set at eval time from the recovering step; empty here
            means "resolved per call".
        system_prompt: Directly readable system prompt.  Must enforce
            "conclusion only, no filler" (primitive-step.md §3).
        user_prompt: Directly readable template over the recovered context.
            ``{context}`` is replaced with the backtracker's ``context_rebuild``;
            ``{objective}`` / ``{step_id}`` are replaced when provided.
        tools: Read-only projection names only (``LivePolicy`` /
            ``resolve_offloaded``).  Empty for the built-ins — no domain writes.
    """

    name: str
    step_type: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    tools: tuple[str, ...] = field(default_factory=tuple)

    def render_user_prompt(
        self,
        *,
        context: str,
        objective: str = "",
        step_id: str = "",
    ) -> str:
        """Render the user prompt over the recovered context."""
        return self.user_prompt.format(
            context=context or "(no recovered context)",
            objective=objective or "(none)",
            step_id=step_id or "(unknown)",
        )

    def eval(
        self,
        *,
        context: str,
        backend: Any,
        step_type: str = "",
        objective: str = "",
        step_id: str = "",
    ) -> str:
        """Evaluate the inline step as a one-shot LLM call (no tools, no session).

        Returns the clean next-action text.  ``backend`` must implement
        ``IRuntimeBackend.complete``; a missing backend raises
        ``InlineStepEvalError`` (the act cannot reason without an LLM — it has
        no kernel session).
        """
        if backend is None:
            raise InlineStepEvalError(
                f"inline step '{self.name}' needs an LLM backend "
                "(one-shot call, no kernel session); none was provided"
            )
        prompt = self.render_user_prompt(
            context=context, objective=objective, step_id=step_id
        )
        try:
            return backend.complete(prompt, system=self.system_prompt).strip()
        except Exception as exc:  # noqa: BLE001 - re-raise with inline-step context
            raise InlineStepEvalError(
                f"inline step '{self.name}' eval failed: {type(exc).__name__}: {exc}"
            ) from exc


class InlineStepEvalError(RuntimeError):
    """The inline step could not produce a next-action (missing backend / LLM)."""


class InlineStepRegistry:
    """Framework registry of inline-step types, keyed by name."""

    def __init__(self, inline_steps: list[InlineStep] | None = None) -> None:
        self._inline_steps: dict[str, InlineStep] = {}
        for inline_step in inline_steps or []:
            self.register(inline_step)

    def register(self, inline_step: InlineStep) -> None:
        """Register *inline_step* by name (overwrites an existing name)."""
        self._inline_steps[inline_step.name] = inline_step

    def get(self, name: str) -> InlineStep | None:
        """Return the inline step named *name*, or None."""
        return self._inline_steps.get(name)

    def has(self, name: str) -> bool:
        """Return True when *name* is a registered inline step."""
        return name in self._inline_steps

    def names(self) -> list[str]:
        """Return all registered inline-step names (insertion order)."""
        return list(self._inline_steps)


# ---------------------------------------------------------------------------
# Built-in compaction acts (framework-owned, registered by name)
# ---------------------------------------------------------------------------

_BACKTRACK_PLAN_SYSTEM = (
    "You are a recovery planner inlined inside a resuming step. Read the "
    "recovered round context and produce ONE concrete next-step instruction "
    "for the recovering step. Output ONLY the instruction text — no preamble, "
    "no 'based on…', no 'I infer…', no reasoning, no JSON, no markdown fences."
)

_BACKTRACK_PLAN_USER = (
    "The following is the recovered context of an interrupted round "
    "(step '{step_id}', objective: {objective}):\n\n"
    "{context}\n\n"
    "What should the recovering step do next? Give one concrete, "
    "self-contained instruction."
)

_NEXT_STEP_SYSTEM = (
    "You are a next-step advisor inlined inside a resuming step. Produce ONE "
    "concrete next-step instruction for the recovering step. Output ONLY the "
    "instruction text — no preamble, no reasoning, no JSON, no markdown fences."
)

_NEXT_STEP_USER = (
    "Recovered context for step '{step_id}' (objective: {objective}):\n\n"
    "{context}\n\n"
    "What should this step do next? Give one concrete, self-contained "
    "instruction."
)


def backtrack_plan_compaction() -> InlineStep:
    """The ``backtrack_plan`` compaction — compacts the backtracker rebuild into
    a next-step instruction (primitive-step.md §1)."""
    return InlineStep(
        name="backtrack_plan",
        system_prompt=_BACKTRACK_PLAN_SYSTEM,
        user_prompt=_BACKTRACK_PLAN_USER,
    )


def next_step_compaction() -> InlineStep:
    """The ``next_step`` compaction — a lighter next-step compaction."""
    return InlineStep(
        name="next_step",
        system_prompt=_NEXT_STEP_SYSTEM,
        user_prompt=_NEXT_STEP_USER,
    )


_DEFAULT_REGISTRY: InlineStepRegistry | None = None


def default_registry() -> InlineStepRegistry:
    """Return the framework registry with the built-in compaction acts registered."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InlineStepRegistry(
            [backtrack_plan_compaction(), next_step_compaction()]
        )
    return _DEFAULT_REGISTRY


def get_inline_step(name: str) -> InlineStep | None:
    """Resolve an inline step by name from the framework registry."""
    return default_registry().get(name)
