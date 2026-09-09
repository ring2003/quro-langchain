"""Steering — progressive re-calibration of exploration frontier (Steering-Semantics.md).

Steering selects the next Exploration Frontier (unresolved region of investigation)
and materializes it as a NextInstruction for the StepAgent.  It is NOT objective
decomposition into task chains (obj_0 → obj_1 → ... → obj_n).

Instead, it maintains:
  - Seed Objective (stable) — the overall intent
  - Exploration Frontier (evolves) — current unresolved WHERE
  - NextInstruction.focus — execution-facing representation of the frontier

The loop operates as:
  1. Steering selects the next frontier from current understanding
  2. Steering creates a NextInstruction (focus = frontier representation)
  3. StepAgent executes the frontier, producing evidence
  4. Steering observes evidence and re-calibrates (back to step 1)

This module ships the **mechanism** only — the loop + the one-shot act.
``explore`` is a composite step built *from* this mechanism.

Design invariants (Steering-Semantics.md §3–8):

- **Frontier selection** — steering chooses WHERE next; it does NOT decompose
  seed into a predetermined task list. The frontier evolves after each cycle
  based on accumulated evidence (understanding).
- **Seed stability** — the seed objective stays constant; the frontier changes.
  Do NOT rewrite the seed.
- **NextInstruction.focus is frontier** — the focus field is the frontier's
  execution-facing representation, never a replacement seed objective.
- **StepAgent freedom** — the agent decides HOW (tool/file selection, concrete
  exploration method) once the frontier is selected.
- **Time-disjoint from Compaction** — compaction retries the same frontier
  (same WHERE, recover HOW); steering selects a new frontier (WHERE changes).
  ``retry_budget`` is the boundary: how many times compaction may retry before
  steering may replace the frontier.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quro.runtime.base import IRuntimeBackend


class SteeringError(RuntimeError):
    """The steering act could not produce the next objective."""


@dataclass(frozen=True)
class ContextView:
    """Unified context for steering and step-agent prompts.

    own_history: This step's own prior L1 round artifacts (temporal).
    dependencies: ACL-checked depends_on predecessor artifacts (spatial).
    """
    own_history: list[dict]  # Artifact summaries from prior rounds
    dependencies: list[dict]  # Artifact summaries from depends_on steps


def build_context_view(state: dict, step_id: str) -> ContextView:
    """Build a ContextView from session state."""
    all_artifacts = state.get("artifacts", []) if isinstance(state, dict) else []

    # Temporal: this step's own prior artifacts
    own = [a for a in all_artifacts if a.get("step_id") == step_id]

    # Spatial: depends_on predecessors with access=="hints"
    step_entry = None
    for s in state.get("steps", []):
        if s.get("step_id") == step_id:
            step_entry = s
            break

    deps = []
    if step_entry and step_entry.get("access") == "hints":
        dep_ids = set(step_entry.get("depends_on", []))
        deps = [a for a in all_artifacts if a.get("step_id") in dep_ids]

    return ContextView(own_history=own, dependencies=deps)


def _artifacts_detail(state: Any) -> str:
    """Render the steering act's artifact view via the ``steering_artifacts`` block.

    Single source of artifact detail for the act's user prompt: the block
    carries FULL artifact bodies (own history + dependencies) rendered from
    the raw session state.  No raw state (e.g. direct unit-test calls) →
    empty view.
    """
    if not state:
        return ""
    from quro.context.blocks import render_block

    return render_block(
        "steering_artifacts", state,
        phase="", budget=0, principal="steering",
    )


# ---------------------------------------------------------------------------
# NextInstruction — structured output from the steering act
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NextInstruction:
    """Structured execution-facing representation of the current Exploration Frontier.

    Emitted by the steering act each round, this bridges the Frontier (semantic
    concept) to the StepAgent (executor).  The act never replaces the seed
    objective; it only selects WHERE to explore next.

    Attributes:
        round_index: The steering cycle this instruction targets (0-based).
        focus: The current frontier's execution-facing representation.
            Narrow and self-contained, never repeats the seed objective.
            This is WHERE the StepAgent should focus (discovery, tracing, etc.),
            not HOW to do it.
        scope_in: Concrete anchors/entry points (symbols, files, modules).
            Useful starting points; NOT a mandatory checklist.
        scope_out: Explicit exclusions (already covered, known dead ends).
            Relevant discovered connections remain traversable.
        recall: Recall tool IDs (artifact references for the step agent to
            address by id rather than guessing from opaque counts).
        done_when: Local completion predicate for this frontier.
            Resolution guidance for the current frontier, evaluated by the
            steering act itself against the artifact evidence it receives;
            distinct from the step's own ``is_complete``. Informs Steering
            (not StepAgent) about when to move to the next frontier.
        is_final_round: The steering act's declaration that the seed objective
            is satisfied. This is Steering's exit token (Steering-Semantics.md
            §13): when the act emits it, the loop exits after the current cycle.
            It is produced by the act from evidence (its own ``done_when``
            checked against the ``steering_artifacts`` view), journaled for
            audit, and Steering-owned — the step-agent's L0 tools can never
            set it. The only other exit is the round budget (Steering-Semantics.md §15).
    """

    round_index: int
    focus: str
    scope_in: list[str] = field(default_factory=list)
    scope_out: list[str] = field(default_factory=list)
    recall: list[str] = field(default_factory=list)
    done_when: str = ""
    is_final_round: bool = False

    @property
    def objective(self) -> str:
        """Backward-compat alias: the focus text as a plain string.

        Used by SteeringLoop.drive and adapter._act where the old code
        expected a plain objective string.
        """
        return self.focus

    def render_for_step_agent(self) -> str:
        """Render the canonical ``next_instruction`` block (kind=31, §9.2).

        This is the ONE place this text is built. Both
        ``session_journal.py`` (audit) and the step-agent's prompt
        (``NextInstructionBlockHook``) should call this rather than each
        re-formatting the fields independently — that duplication is the
        "shadow implementation" pattern this framework treats as a named
        defect (see project memory / bug-remediation notes). Empty
        sections are omitted so the block stays short and unambiguous
        rather than padded with "(none)" noise.
        """
        lines = [f"FOCUS: {self.focus}"]
        if self.scope_in:
            lines.append(f"SCOPE_IN: {', '.join(self.scope_in)}")
        if self.scope_out:
            lines.append(f"SCOPE_OUT: {', '.join(self.scope_out)}")
        if self.recall:
            lines.append(f"RECALL: {', '.join(self.recall)}")
        lines.append(f"DONE_WHEN: {self.done_when or '(use judgment)'}")
        return "\n".join(lines)


# The one archive token. There is exactly one accepted spelling and exactly
# one accepted place for it (the SIGNAL field, matched as a full field
# value) — never a substring search over the whole raw completion. Substring
# search was the actual bug: the token could appear incidentally inside
# SCOPE_OUT/DONE_WHEN prose (e.g. "excluding files already marked
# STEP_OBJECTIVE_ARCHIVED in round 2") and would falsely end the loop, and it
# duplicated / could disagree with a separate FINAL: yes|true|1 field.
_ARCHIVE_SIGNAL = "STEP_OBJECTIVE_ARCHIVED"

_FIELD_RE = re.compile(
    r"^(FOCUS|SCOPE_IN|SCOPE_OUT|RECALL|DONE_WHEN|SIGNAL)\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _extract_signal(fields: dict[str, str], raw: str) -> bool:
    """Return True iff the model emitted the exact archive token.

    Single source of truth, exact match only:
      1. a ``SIGNAL:`` field whose value equals the token exactly, or
      2. (unstructured fallback) the *entire* raw output, stripped, equals
         the token exactly — not "contains".

    Anything else (no SIGNAL field, or the token embedded in other prose)
    is NOT final. The result feeds ``NextInstruction.is_final_round`` —
    Steering's own exit token (Steering-Semantics.md §13): it is journaled
    for audit AND governs the loop's exit when True.
    """
    value = fields.get("SIGNAL", "").strip().strip("\"'")
    if value.upper() == _ARCHIVE_SIGNAL:
        return True
    if not fields and raw.strip().upper() == _ARCHIVE_SIGNAL:
        return True
    return False


def _parse_next_instruction(raw: str, *, round_index: int) -> NextInstruction:
    """Parse structured LLM output into a ``NextInstruction``.

    Expected format (each line is ``KEY: value``)::

        FOCUS: <text>
        SCOPE_IN: <item1>, <item2>
        SCOPE_OUT: <item1>, <item2>
        RECALL: <item1>, <item2>
        DONE_WHEN: <text>
        SIGNAL: STEP_OBJECTIVE_ARCHIVED     # omit unless truly done

    Falls back gracefully: missing keys get defaults, unknown keys are
    ignored.  If the output is plain text (no structured fields detected
    at all), it is treated as the focus field with all other fields empty
    and ``is_final_round`` only True if the *entire* output is exactly the
    archive token (see ``_extract_signal``).
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    fields: dict[str, str] = {}
    for line in lines:
        m = _FIELD_RE.match(line)
        if m:
            fields[m.group(1).upper()] = m.group(2).strip()

    if not fields:
        return NextInstruction(
            round_index=round_index,
            focus=raw.strip(),
            is_final_round=_extract_signal(fields, raw),
        )

    return NextInstruction(
        round_index=round_index,
        focus=fields.get("FOCUS", ""),
        scope_in=_split_csv(fields.get("SCOPE_IN", "")),
        scope_out=_split_csv(fields.get("SCOPE_OUT", "")),
        recall=_split_csv(fields.get("RECALL", "")),
        done_when=fields.get("DONE_WHEN", ""),
        is_final_round=_extract_signal(fields, raw),
    )


def _split_csv(text: str) -> list[str]:
    """Split a comma-separated value string, stripping whitespace."""
    if not text or not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


@runtime_checkable
class ISteeringAct(Protocol):
    """Select the next Exploration Frontier given a bounded state view.

    The act is a one-shot LLM call — no kernel session, no domain write tools.
    Its product is a ``NextInstruction`` (structured) representing the next
    frontier's execution-facing form.

    This is a re-calibration operation: the act observes the current understanding
    (state_view + artifacts) and selects WHERE to investigate next. It does NOT
    decompose the seed objective into task chains.
    """

    def next_objective(
        self,
        *,
        state_view: str,
        objective: str,
        step_objective: str,
        expected_output: str,
        prior_objectives: list[str],
        round_index: int,
        backend: IRuntimeBackend,
        hint: str = "",
        state: Any | None = None,
    ) -> NextInstruction | str:
        """Return the next frontier instruction (structured) or text (legacy).

        Args:
            state_view: Bounded snapshot of current investigation state.
            objective: The current frontier focus.
            step_objective: The step's seed objective — the anchor the act uses
                to stay aligned with the step's true goal. Stable across cycles.
            expected_output: What the step should ultimately produce.
            prior_objectives: Frontiers from all prior cycles (history).
            round_index: Current steering cycle index (0-based).
            backend: The LLM backend for the one-shot call.
            hint: Sufficiency hint from the gate (read-only fact, not exit signal).
            state: Raw session state dict carrying the step's artifacts. When
                provided, the act's artifact view is rendered from it via the
                ``steering_artifacts`` block (FULL content, not a summary
                index); without it the artifact view is empty.
        """
        ...


@runtime_checkable
class ISteeringGate(Protocol):
    """Sufficiency hints and budget cap for the steering loop.

    Kept protocol-driven (blueprint §protocol-driven): the loop depends on this
    protocol, never on a concrete domain state shape. A domain declares its own
    budget logic instead of the framework hard-coding one.

    The gate does NOT decide frontier completion: the loop exits only when the
    steering act declares a final round (``NextInstruction.is_final_round``)
    or when the declared round budget is exhausted (Steering-Semantics.md §15).
    """

    def suggested_sufficiency(self, state: Any, *, objective: str) -> str:
        """FACT layer: a hint string fed into the steering act's prompt.

        Read-only signal — NEVER consulted by drive() to end the loop.
        Example: '3 artifacts collected for this frontier, minimum expected is 1.'
        """
        ...

    def budget_exhausted(self, state: Any, *, rounds: int) -> bool:
        """Return True when the steering loop must stop (round budget reached)."""
        ...


@runtime_checkable
class ISteeringFolder(Protocol):
    """Fold the loop's ``N`` intermediate artifacts into the step's one artifact."""

    def fold(self, state: Any, *, artifacts: list[Any]) -> Any:
        """Return the single folded artifact (the step's terminal artifact)."""
        ...


@dataclass
class SteeringConfig:
    """Steering loop configuration.

    Args:
        max_rounds: Hard round cap (each round = one objective -> one act ->
            one artifact).  Guards against a runaway loop; ``max_rounds=0``
            disables the cap.
        retry_budget: How many times compaction may retry the *same* objective
            (within-objective recovery) before control returns to steering
            (between-objective).  ``0`` = no compaction retries: an
            unsuccessful act hands straight back to steering.  This is the
            time-disjoint boundary value the domain must declare
            (primitive-step.md §7) — exploration domains set it small,
            hard-crack domains set it large.
    """

    max_rounds: int = 12
    retry_budget: int = 3


@dataclass
class SteeringDriveResult:
    """Outcome of :meth:`SteeringLoop.drive`.

    Attributes:
        rounds: The per-round records.
        exit_reason: Why the loop stopped — ``"final_round"`` (the steering
            act declared the final round) or ``"budget_exhausted"`` (the
            declared round cap was reached). Never L0-owned: ``complete_step``
            and ``round_status`` play no part in it.
    """

    rounds: list["SteeringRound"]
    exit_reason: str = ""


@dataclass
class SteeringRound:
    """One steering round: an objective override and its act outcome."""

    objective: str
    artifact: Any = None
    success: bool = False
    next_instruction: NextInstruction | None = None


_DEFAULT_STEERING_SYSTEM_PROMPT = (
    "You are the steering mechanism inside one composite step that explores "
    "progressively by re-calibrating its frontier.\n\n"
    "Your job is NOT to decompose the seed objective into a task list "
    "(obj_0 → obj_1 → ... → obj_n). Instead, you select the NEXT FRONTIER "
    "(most valuable unresolved region) given the current evidence.\n\n"
    "The seed objective (overall intent) stays stable. The frontier (WHERE to "
    "investigate next) evolves after each cycle of exploration.\n\n"
    "Output EXACTLY these fields, one per line, nothing else — no preamble, "
    "no reasoning, no JSON, no markdown fences:\n\n"
    "FOCUS: <this cycle's frontier — never restate the seed objective, never "
    "repeat prior frontiers. Narrow, self-contained, describes unresolved knowledge>\n"
    "SCOPE_IN: <comma-separated entry-point anchors: symbols, files, modules>\n"
    "SCOPE_OUT: <comma-separated exclusions: already covered, dead ends>\n"
    "RECALL: <comma-separated recall tool IDs, if needed>\n"
    "DONE_WHEN: <a checkable predicate over artifact content — when this holds, "
    "this frontier is resolved>\n"
    "SIGNAL: <omit this line normally. Include it ONLY as the literal "
    "'STEP_OBJECTIVE_ARCHIVED' when you believe the seed objective is now satisfied>\n\n"
    "How your SIGNAL is used (read this so you can propose with confidence): "
    "SIGNAL is your proposal, journaled for audit — it is NOT the exit switch. "
    "The framework independently checks your DONE_WHEN against the actual "
    "artifacts (this is the fact layer; it never trusts your wording) and only "
    "that check ends the loop. A round budget is also enforced as a backstop. "
    "So: propose SIGNAL whenever you genuinely believe the work is done — "
    "being wrong costs nothing, the framework will simply continue.\n\n"
    "Rules:\n"
    "- FOCUS describes unresolved knowledge, not a prescribed execution sequence.\n"
    "- FOCUS is narrow; never restate the seed objective or a prior frontier.\n"
    "- Stay aligned with the seed objective — narrow toward it, don't drift.\n"
    "- Use the artifacts listed below by their id — steer toward gaps in them, "
    "not around an opaque count.\n"
    "- A frontier constrains attention, not actions: the agent may discover "
    "connected regions during exploration."
)

_DEFAULT_STEERING_USER_PROMPT = (
    "SEED OBJECTIVE (stable intent, do NOT change or restate in FOCUS):\n{step_objective}\n\n"
    "Expected output: {expected_output}\n\n"
    "Current Frontier (WHERE to explore next, not HOW): {objective}\n"
    "Cycle: {round_index}\n\n"
    "Prior Frontiers (history of WHERE we explored):\n{prior_objectives}\n\n"
    "Artifacts produced so far (steer toward gaps in these, use their ids):\n"
    "{artifacts_detail}\n\n"
    "Progress notes (observations from this cycle):\n{state_view}\n\n"
    "What is the next frontier? Output the fields only."
)


class SteeringRegistry:
    """Per-name prompt registry for Steering acts.

    Domains register steering names (e.g. ``"narrow"``) with custom
    ``(system_prompt, user_prompt)`` pairs.  When ``Steering`` is
    instantiated via ``resolve_steering``, the registry is consulted
    first; a registered name gets its domain-specific prompts, while
    an unregistered name falls back to the framework defaults.

    Modeled after ``InlineStepRegistry`` (inline_step.py) but for the
    invasive objective-override case.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, str]] = {}

    def register(
        self,
        name: str,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> None:
        """Register a steering name with custom prompts.

        Overwrites an existing entry for *name*.
        """
        self._entries[name] = (system_prompt, user_prompt)

    def get_prompts(self, name: str) -> tuple[str, str] | None:
        """Return ``(system_prompt, user_prompt)`` for *name*, or None."""
        return self._entries.get(name)

    def has(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries)


_DEFAULT_STEERING_REGISTRY: SteeringRegistry | None = None


def default_steering_registry() -> SteeringRegistry:
    """Return the framework-global steering registry (lazy singleton)."""
    global _DEFAULT_STEERING_REGISTRY
    if _DEFAULT_STEERING_REGISTRY is None:
        _DEFAULT_STEERING_REGISTRY = SteeringRegistry()
    return _DEFAULT_STEERING_REGISTRY


def resolve_steering(
    name: str,
    *,
    registry: SteeringRegistry | None = None,
) -> "Steering":
    """Resolve a ``Steering`` instance by name, consulting the registry.

    If *name* is registered, its custom prompts are used; otherwise the
    framework defaults apply.
    """
    reg = registry or default_steering_registry()
    prompts = reg.get_prompts(name)
    if prompts is not None:
        sys_prompt, usr_prompt = prompts
        return Steering(
            name=name,
            system_prompt=sys_prompt or _DEFAULT_STEERING_SYSTEM_PROMPT,
            user_prompt=usr_prompt or _DEFAULT_STEERING_USER_PROMPT,
        )
    return Steering(name=name)


class Steering:
    """The steering act: override the objective given a bounded state view.

    A thin prompt over the running step's state.  The prompt enforces "objective
    text only, no reasoning", the same discipline as the compaction built-ins
    (inline_step.py) but for the *invasive* objective-override case.

    When ``coordinator_hooks`` is provided, the prompt is assembled via
    ``PromptCoordinator`` (block-attributed, same as the governor path).
    Otherwise the legacy hand-built prompt is used.
    """

    def __init__(
        self,
        *,
        name: str = "steering",
        system_prompt: str = "",
        user_prompt: str = "",
        coordinator_hooks: list[Any] | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt or _DEFAULT_STEERING_SYSTEM_PROMPT
        self.user_prompt = user_prompt or _DEFAULT_STEERING_USER_PROMPT
        self._coordinator_hooks = coordinator_hooks

    def render_user_prompt(
        self,
        *,
        state_view: str,
        objective: str,
        step_objective: str = "",
        expected_output: str = "",
        prior_objectives: list[str] | None = None,
        round_index: int = 0,
        hint: str = "",
        state: Any | None = None,
    ) -> str:
        prior = "\n".join(
            f"  {i+1}. {obj}" for i, obj in enumerate(prior_objectives or [])
        ) or "  (none yet)"
        prompt = self.user_prompt.format(
            step_objective=step_objective or "(none)",
            expected_output=expected_output or "(none)",
            objective=objective or "(none)",
            prior_objectives=prior,
            state_view=state_view or "(no progress)",
            round_index=round_index,
            artifacts_detail=_artifacts_detail(state),
        )
        if hint:
            prompt += f"\nSufficiency hint: {hint}"
        return prompt

    def next_objective(
        self,
        *,
        state_view: str,
        objective: str,
        step_objective: str,
        expected_output: str,
        prior_objectives: list[str],
        round_index: int = 0,
        backend: IRuntimeBackend,
        hint: str = "",
        state: Any | None = None,
    ) -> NextInstruction:
        if backend is None:
            raise SteeringError(
                f"steering act '{self.name}' needs an LLM backend "
                "(one-shot call, no kernel session); none was provided"
            )
        if self._coordinator_hooks is not None:
            system, user = self._assemble_via_coordinator(
                state_view=state_view,
                objective=objective,
                step_objective=step_objective,
                expected_output=expected_output,
                prior_objectives=prior_objectives,
                round_index=round_index,
                hint=hint,
                state=state,
            )
            # Build message list from assembled blocks.
            messages: list[dict[str, Any]] = []
            for content in system.values():
                messages.append({"role": "system", "content": content})
            for content in user.values():
                messages.append({"role": "user", "content": content})
            try:
                result = backend.complete_one(
                    messages, log_prefix="[steering-act] ",
                )
                raw = result.content.strip()
            except Exception as exc:
                raise SteeringError(
                    f"steering act '{self.name}' eval failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        else:
            prompt = self.render_user_prompt(
                state_view=state_view,
                objective=objective,
                step_objective=step_objective,
                expected_output=expected_output,
                prior_objectives=prior_objectives,
                round_index=round_index,
                hint=hint,
                state=state,
            )
            try:
                raw = backend.complete_with_meta(
                    prompt, system=self.system_prompt,
                    log_prefix="[steering-act] ",
                ).content.strip()
            except Exception as exc:
                raise SteeringError(
                    f"steering act '{self.name}' eval failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return _parse_next_instruction(raw, round_index=round_index)

    def _assemble_via_coordinator(
        self,
        *,
        state_view: str,
        objective: str,
        step_objective: str,
        expected_output: str,
        prior_objectives: list[str],
        round_index: int,
        hint: str,
        state: Any | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Assemble system/user blocks via PromptCoordinator.

        Uses the coordinator's block assembly to build the prompt,
        eliminating the shadow prompt-assembly path.  The system block
        comes from the steering identity; user blocks are assembled from
        the coordinator hooks plus dynamic steering-specific blocks.

        The dynamic steering content rides as the ordered ``steering_context``
        block (kind 29, objective band), and the full artifact view rides as
        the ``steering_artifacts`` block rendered from the raw state via
        ``StateBlocksHook`` — both flow through the coordinator's
        ordering/trimming, never post-injected.  (The old
        ``user_blocks[\"steering_context\"] = ...`` seam was a bypass that
        escaped block ordering and trimming.)
        """
        from quro.context.coordinator import PromptCoordinator, SystemBlocksHook
        from quro.context.coordinator import StateBlocksHook

        # Build the user content for this round.
        prior = "\n".join(
            f"  {i+1}. {obj}" for i, obj in enumerate(prior_objectives or [])
        ) or "  (none yet)"
        user_content = self.user_prompt.format(
            step_objective=step_objective or "(none)",
            expected_output=expected_output or "(none)",
            objective=objective or "(none)",
            prior_objectives=prior,
            state_view=state_view or "(no progress)",
            round_index=round_index,
            artifacts_detail=_artifacts_detail(state),
        )
        if hint:
            user_content += f"\nSufficiency hint: {hint}"

        class _SteeringContextHook:
            """Ordered user block carrying the dynamic steering act prompt."""

            name = "steering_context"
            block_role = "user"

            def get_blocks(self, *args: Any, **kwargs: Any) -> dict[str, str]:
                return {"steering_context": user_content}

        # Assemble: system identity + user content via coordinator.  The
        # dynamic steering prompt rides as an ordered ``steering_context``
        # block and the full artifact view as a ``steering_artifacts`` block —
        # both flow through the coordinator's ordering/trimming.  No
        # post-injection: the coordinator owns the final user block order.
        hooks: list[Any] = [
            SystemBlocksHook(self.system_prompt),
            _SteeringContextHook(),
            *self._coordinator_hooks,
        ]
        if state:
            hooks.append(StateBlocksHook(
                ["steering_artifacts"], state,
                phase="", budget=0, principal="steering",
            ))
        coordinator = PromptCoordinator(hooks, principal="steering")
        return coordinator.assemble()


class SteeringLoop:
    """The loop that drives progressive frontier re-calibration inside a composite step.

    The loop is **mechanism-only**: it does not know what ``act`` does or how
    ``fold`` shapes the terminal artifact — both are injected protocols.  A
    domain builds its ``explore`` composite step on top of this loop.

    Semantics (Steering-Semantics.md §8–15):
      - Seed Objective (stable) — the overall intent
      - Exploration Frontier (evolves) — current WHERE, selected by Steering
      - NextInstruction.focus — execution-facing representation of the frontier
      - StepAgent — executor; decides HOW (tool/file selection)
      - Steering Cycle — one frontier selection + one bounded L0 execution episode

    The loop:
        frontier = steer(understanding)           # where next
        artifact = act(frontier, state)           # exploration with possible retries
        repeat until the act declares is_final_round / budget exhausted
        return fold(artifacts)

    Key properties:
      - Compaction (retry-within-frontier) is intra-cycle; only Steering moves
        to the next frontier (Steering-Semantics.md §10).
      - is_final_round is Steering's exit token, not StepAgent's (§13).
      - Artifact/evidence informs Steering but cannot independently terminate
        the loop (§16–17).
    """

    def __init__(
        self,
        *,
        act: ISteeringAct,
        gate: ISteeringGate,
        folder: ISteeringFolder,
        config: SteeringConfig | None = None,
    ) -> None:
        self._act = act
        self._gate = gate
        self._folder = folder
        self._config = config or SteeringConfig()

    # Run is delegated to a runner that owns the live state + backend; the loop
    # itself stays pure w.r.t. those (no hidden mutable members), so it can be
    # replayed.  The runner is provided below as a convenience.
    def drive(
        self,
        *,
        backend: IRuntimeBackend,
        seed_objective: str,
        step_objective: str,
        expected_output: str,
        state_view: Callable[[], str],
        state: Callable[[], Any],
        act: Any,
        journal: Any | None = None,
    ) -> list[SteeringRound]:
        """Drive the loop to completion, returning the per-round records.

        Implements the canonical model (Steering-Semantics.md §20):
          - seed stays stable
          - frontier evolves after each cycle
          - evidence accumulates into understanding
          - steering re-calibrates based on understanding

        ``state_view`` is a callable that returns the current state snapshot
        as a string for the steering act. ``state`` is a callable that returns
        the raw session state dict — passed to the gate so it can inspect
        artifacts, step status, etc., and to the act so its artifact view is
        rendered from raw state via the ``steering_artifacts`` block (FULL
        content, not a summary index). Both are called **each cycle**.

        ``act`` is a callable ``act(frontier, *, signal="", next_instruction=None)
        -> (artifact, success)`` supplied by the caller (the composite step's
        executor); the loop only sequences steering → act → gate/budget.
        The caller owns the live state (updated inside ``act``) and folds the
        artifacts at the end via ``self._folder``.

        ``journal`` is an optional ``IMetaPlannerSessionJournal``.  When
        provided, each cycle is recorded as a ``SteeringRoundEntry`` for
        audit and replay.

        Exit conditions (Steering-Semantics.md §15) — ONLY these two:
          1. the steering act declares the final round, i.e. it produces a
             ``NextInstruction`` with ``is_final_round=True`` (Steering's own
             exit token, §13 — produced by the act from evidence, never settable
             by the step-agent's L0 tools);
          2. ``gate.budget_exhausted(state, rounds=len(rounds))`` returns True
             (round budget exhausted — safety net).

        ``complete_step`` / ``round_status`` are L0-owned facts and must never
        gate the loop's exit (Steering-Semantics.md Anti-pattern E).

        Returns a ``SteeringDriveResult`` carrying the per-round records and
        the ``exit_reason`` (``"final_round"`` | ``"budget_exhausted"``), so the
        caller's fold keys pipeline advancement off the Steering-approved exit
        rather than off L0-owned ``round_status``.
        """
        from quro.planner.session_journal import SteeringRoundEntry

        rounds: list[SteeringRound] = []
        frontier = seed_objective  # Current frontier (initial = seed)
        prior_frontiers: list[str] = []  # History of selected frontiers
        exit_reason = ""
        for cycle_idx in range(self._config.max_rounds or 1):
            # Re-evaluate state_view each cycle so the act sees fresh artifacts
            # and updated understanding.
            current_view = state_view()
            
            # Get sufficiency hint from gate (read-only fact, not exit authority).
            hint = self._gate.suggested_sufficiency(state(), objective=frontier)
            
            # Steering: select next frontier (WHERE to investigate next).
            result = self._act.next_objective(
                state_view=current_view,
                objective=frontier,  # Current frontier for context
                step_objective=step_objective,  # Seed (stable)
                expected_output=expected_output,
                prior_objectives=prior_frontiers,  # Frontier history
                round_index=cycle_idx,
                backend=backend,
                hint=hint,
                state=state(),
            )
            
            # Normalize: handle both NextInstruction and legacy str.
            # The act always returns NextInstruction or str;
            # we normalize to NextInstruction.
            if isinstance(result, NextInstruction):
                ni = result
                frontier = ni.focus  # Extract the frontier representation
            else:
                # Legacy str path: parse into NextInstruction.
                frontier = result
                ni = _parse_next_instruction(result, round_index=cycle_idx)

            # Forward the NextInstruction to the StepAgent for THIS cycle.
            # The focus field carries the frontier representation.
            # Each cycle gets a fresh instruction (not cached from prior round).
            artifact, success = act(
                frontier,
                signal="",
                next_instruction=ni,
            )
            rounds.append(
                SteeringRound(
                    objective=frontier,
                    artifact=artifact,
                    success=success,
                    next_instruction=ni if isinstance(result, NextInstruction) else None,
                )
            )
            prior_frontiers.append(frontier)

            # Journal: record this cycle's entry.
            if journal is not None:
                journal.record_steering_round(SteeringRoundEntry(
                    round_index=cycle_idx,
                    objective=frontier,
                    raw_output=str(result) if not isinstance(result, NextInstruction) else "",
                    next_instruction=ni if isinstance(result, NextInstruction) else None,
                    state_view_snapshot=current_view[:500],
                    gate_hint=hint,
                ))

            # Exit conditions (Steering-Semantics.md §15) — ONLY these two:
            # 1. The steering act declared the final round (is_final_round —
            #    Steering's own exit token, §13). It is a structured token the
            #    act produced from evidence (its own done_when checked against
            #    the artifact view), journaled above for audit.
            # 2. Round budget exhausted (safety net).
            # complete_step / round_status (L0-owned facts) are NEVER read
            # here (Steering-Semantics.md Anti-pattern E).
            if ni.is_final_round:
                # Final round declared: exit the steering loop.
                if journal is not None and rounds:
                    entry = SteeringRoundEntry(
                        round_index=cycle_idx,
                        objective=frontier,
                        exit_reason="final_round",
                    )
                    journal.record_steering_round(entry)
                exit_reason = "final_round"
                break
            if self._gate.budget_exhausted(state(), rounds=len(rounds)):
                # Round budget exhausted: exit the steering loop.
                if journal is not None and rounds:
                    entry = SteeringRoundEntry(
                        round_index=cycle_idx,
                        objective=frontier,
                        exit_reason="budget_exhausted",
                    )
                    journal.record_steering_round(entry)
                exit_reason = "budget_exhausted"
                break  # Safety net, same shape as meta-planner's max_rounds
        return SteeringDriveResult(rounds=rounds, exit_reason=exit_reason)