"""MetaPlanner prompt manager — block-based prompt assembly.

Splits the system prompt into composable blocks that can be dynamically
assembled, reordered, and budget-trimmed.

Block priority determines inclusion order when a token budget is applied:
lower priority numbers are kept first; higher numbers are trimmed first.

Priority bands (see meta-plan-prompt-and-ppf.md §4):
    1-9   REQUIRED — never trimmed. 1-3 = static system instructions.
          6-9 = dynamic per-round content the design doc names as
          hard-retained (current round's actual outcome, the plan gate,
          and the problem/plan skeleton that "stay in view" per §1).
    10+   WEIGHTED — trimmable under budget pressure. Ordering among
          these is soft; only their presence, not their survival, is
          expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ============================================================================
# PromptBlock — a single composable section of the system prompt
# ============================================================================


@dataclass(frozen=True)
class PromptBlock:
    """A single composable section of the system prompt.

    Attributes:
        name: Unique block identifier (e.g. ``"identity"``).
        role: ``"system"`` for identity/workflow blocks, ``"user"`` for
              context/examples/rules that sit at the end.
        priority: Inclusion priority — lower numbers are kept first when
                  budget trimming is applied. Priorities < 10 are never
                  trimmed (see module docstring); 10+ is the weighted,
                  trimmable tier.
        content: The rendered text content of this block.
    """

    name: str
    role: str
    priority: int
    content: str


# ============================================================================
# Static block content
#
# Consolidated from 7 overlapping sections (identity / plan_gate /
# g_f_workflow / mental_model / tool_ref / examples / rules) down to 4.
# Each fact below — no verdict, plan gate mechanics, create_step contract,
# run_custom_pipeline authority, unique step_ids, audit-before-replan — was
# previously stated 2-4 times across sections; it is now stated once, in
# the block where an LLM would look for it.
# ============================================================================

_IDENTITY = """You are the MetaPlanner: a planner and auditor, not an executor,
driving one round at a time of a codebase-research scratchpad loop
(survey → narrow → deep-dive → synthesize → report).

The framework owns g/f separation. Each round you write `StepSpec`s
directly (`g`, via `create_step`); the framework executes them
(`f`, via `run_custom_pipeline`) into Artifacts and computes `goal_status`
(sat/unsat facts) itself, via its own `evaluate`. You never execute a step.

There is no verdict. `goal_status` is a framework-computed fact you read,
never write. Your `stop("...")` text is a statement for the user ("in my
view this is solved"), not a control token — nothing you write, including
the words SAT/UNSAT, is interpreted as a signal. You alone decide, from the
facts, when to keep going or stop."""

_WORKFLOW = """## Each round — two required acts, in order

**1. Build & run (you perform `g`)**
- `create_step(step_id, objective, step_type, depends_on, skills)` — declare
  one narrowly-scoped step. `step_id` unique for the whole session (not just
  this round — e.g. `r3_dive_deep_a`); `depends_on` is a comma-separated list
  of step_ids for ordering; `step_type` must be registered (`list_operators`);
  `skills` are validated against that step type's pool.
- `run_custom_pipeline()` — runs every step you've declared this round, in
  dependency order. This is `f`, and its result is the only authoritative
  round outcome — never substitute your own judgment for it.
- `get_artifact(artifact_id)` — inspect a result before deciding what's next.
- Nothing new needed this round (last round's steps already cover it)?
  Skip straight to step 2.
- `run_pipeline` / `build_pipeline` / `define_method` / `list_methods` reach
  the HTN solver, for verification only — never treat their output as the
  round's real outcome.

**2. Gate the round**
- `submit_plan("<text>")` — required to open another round. This is a hard
  gate: no next round without it. A verbatim resubmission ("keep this plan")
  is valid and still journaled as its own decision.
- `stop("<optional conclusion>")` — stopping is free, no plan required.
- End every round with exactly one of the two — and only after any
  building/running is done; neither call executes anything itself.

When a branch has stalled, fix it by revising steps and resubmitting a plan
— never by trying to execute anything yourself.

## Tools

| Tool | Purpose |
|---|---|
| `create_step` | declare a step (primary planning tool) |
| `setup_step` | edit an existing step's objective/dependencies |
| `list_manual_steps` | list steps created this session |
| `run_custom_pipeline` | execute declared steps — the authoritative outcome |
| `get_artifact` | read a step's artifact |
| `submit_plan` / `stop` | the round gate (§2 above) |
| `record_decision` | log why you chose this plan, for later rounds to revisit |
| `commit_learning` / `recall_memory` | persist / retrieve reusable cross-project patterns (only when proven, not per-round) |
| `run_pipeline` / `build_pipeline` / `define_method` / `list_methods` | HTN solver — verification only |
| `list_operators` | query registered step types |"""

_RULES = """## Rules

1. You plan, the framework executes — never do the exploration yourself.
2. One concern per round: narrow across rounds rather than front-loading
   everything into a single plan.
3. Audit before replanning — read the round's actual (`goal_status` +
   artifacts), not your memory of what you asked for.
4. Declare `depends_on` explicitly; an unordered step may run out of order.
   **Why this matters — the invisible-dependency trap (Anti-Pattern I):** the
   framework carries *nothing* implicitly across steps. A step's artifacts are
   invisible to any step that doesn't name it in `depends_on`, and
   `access="hints"` is required *on top of* `depends_on` to actually read their
   bodies. If your objective tells the next step to "read art_ab12cd34" (or to
   go deeper on a file from a prior round) but you didn't declare that
   `depends_on` edge + `access="hints"`, the referenced artifact never enters
   the carried state and your instruction is meaningless to the executor. When
   you reference a prior artifact by id, make sure that id is reachable from
   this step's `depends_on` chain and that the predecessor has
   `access="hints"` — otherwise declare the edge/permission, don't just mention
   the id in prose.
5. Cover every step type the goal needs — an omitted one leaves that goal
   fact unsat forever, since that work simply never runs.
6. Don't invent a `meta-planner` step type.
7. Nobody reads "SAT"/"UNSAT" in your text as a signal — stop only because
   the facts convinced you the problem is solved."""

_EXAMPLE = """## Example

Round 1 — survey then narrow:
```
create_step("survey_modules", "Survey repo layout, list modules relevant to X",
             step_type="survey_module", skills="scan")
create_step("narrow_focus", "Narrow to the two modules most related to X",
             step_type="dive_deep", depends_on="survey_modules")
run_custom_pipeline()
submit_plan("Round 1: broad survey, then narrow to two modules.")
```

Round 3 — a branch stalled; retarget instead of rerunning:
```
get_artifact("art_<deep_dive_artifact>")
create_step("r3_dive_retarget", "Deep-dive <other_module> for evidence",
             step_type="dive_deep", depends_on="r2_survey")
run_custom_pipeline()
submit_plan("Round 3: prior module had no evidence; retargeting to <other_module>.")
```
Continue revising until every goal fact is sat, then `stop` with your
conclusion."""


# ============================================================================
# MetaPlannerPromptManager
# ============================================================================


class MetaPlannerPromptManager:
    """Block-based prompt assembly for the MetaPlanner system prompt.

    Splits the prompt into composable ``PromptBlock`` instances with
    priority-based budget trimming.  Priorities < 10 are never trimmed
    (system instructions *and* the dynamic blocks the design doc calls
    hard-retained: current-round actual, the plan gate, and the
    problem/plan skeleton that must "stay in view"). 10+ is trimmed first,
    highest number first.

    Usage::

        manager = MetaPlannerPromptManager()
        extra = [
            manager.build_ppf_problem_block(problem),
            manager.build_ppf_goals_block(goals),
            manager.build_ppf_plan_skeleton_block(skeleton),
            manager.build_current_round_actual_block(round_idx, sat, unsat, artifacts),
            manager.build_round_context_block(round_idx, plan_text),
            manager.build_ppf_constraints_block(constraints),   # weighted
            manager.build_rounds_trace_block(prior_rounds),     # weighted, historical only
        ]
        prompt = manager.assemble(extra_blocks=extra, budget=8000)
    """

    def __init__(self) -> None:
        self._static_blocks: list[PromptBlock] = [
            PromptBlock(name="identity", role="system", priority=1, content=_IDENTITY),
            PromptBlock(name="workflow", role="system", priority=2, content=_WORKFLOW),
            PromptBlock(name="rules", role="system", priority=3, content=_RULES),
            PromptBlock(name="example", role="user", priority=10, content=_EXAMPLE),
        ]

    def _collect_blocks(
        self,
        available_tools: list[str],
        catalog_names: list[str],
    ) -> list[PromptBlock]:
        """Collect all blocks including dynamic ones.

        Dynamic blocks (ppf_context, current_round_actual, rounds_trace,
        ...) are added by the caller via ``assemble()``'s ``extra_blocks``.
        """
        return list(self._static_blocks)

    def _order(self, blocks: list[PromptBlock]) -> list[PromptBlock]:
        """Sort blocks by priority, then by original order for same priority."""
        return sorted(blocks, key=lambda b: (b.priority, self._block_index(b.name)))

    def _block_index(self, name: str) -> int:
        """Return the original index of a static block by name."""
        for i, b in enumerate(self._static_blocks):
            if b.name == name:
                return i
        return 999

    def _trim(
        self, blocks: list[PromptBlock], budget: int
    ) -> list[PromptBlock]:
        """Remove highest-priority-number blocks until within budget.

        Blocks with priority < 10 are never trimmed — this covers both the
        static system blocks and any dynamic block the caller marked
        required (current-round actual, plan gate context, PPF
        problem/goals/plan). Only the weighted tier (priority >= 10) is
        eligible for trimming.
        """
        if budget <= 0:
            return blocks

        # Estimate token count (rough: 1 token ≈ 4 chars).
        def _estimate_tokens(b: PromptBlock) -> int:
            return len(b.content) // 4

        total = sum(_estimate_tokens(b) for b in blocks)
        if total <= budget:
            return blocks

        # Trim from highest priority number downward.
        trimmed: list[PromptBlock] = []
        for b in reversed(blocks):
            if total <= budget:
                # Remaining blocks fit — prepend and keep.
                trimmed.insert(0, b)
            elif b.priority >= 10:
                # Weighted-tier block — safe to trim.
                total -= _estimate_tokens(b)
            else:
                # Required block (priority < 10) — keep even if over budget.
                trimmed.insert(0, b)
                total -= _estimate_tokens(b)

        return trimmed

    def _render(self, blocks: list[PromptBlock]) -> str:
        """Render ordered blocks into a single prompt string."""
        parts: list[str] = []
        for b in blocks:
            parts.append(b.content)
        return "\n\n".join(parts)

    def assemble(
        self,
        *,
        available_tools: list[str] | None = None,
        catalog_names: list[str] | None = None,
        budget: int = 0,
        extra_blocks: list[PromptBlock] | None = None,
    ) -> str:
        """Assemble the system prompt from blocks.

        Args:
            available_tools: Tool names available in this session (for
                future dynamic tool-ref blocks).
            catalog_names: Step type names in the catalog (for future
                dynamic operator-ref blocks).
            budget: Maximum token count (0 = no trimming).
            extra_blocks: Additional blocks to inject — PPF context,
                current-round actual, round context, historical rounds,
                etc. These are appended after the static blocks and then
                sorted into position by priority.

        Returns:
            The assembled system prompt string.
        """
        blocks = self._collect_blocks(
            available_tools or [],
            catalog_names or [],
        )
        if extra_blocks:
            blocks.extend(extra_blocks)
        ordered = self._order(blocks)
        if budget > 0:
            ordered = self._trim(ordered, budget)
        return self._render(ordered)

    # --------------------------------------------------------- PPF builders
    # Required — priority < 10. Per meta-plan-prompt-and-ppf.md §1, the
    # framework's job is to "keep the problem + plan in view while
    # compressing everything else"; these are that view.

    @staticmethod
    def build_ppf_problem_block(problem: str) -> PromptBlock:
        """Build the PPF problem block (required, never trimmed)."""
        return PromptBlock(
            name="ppf_problem",
            role="user",
            priority=6,
            content=f"**Problem:** {problem}",
        )

    @staticmethod
    def build_ppf_goals_block(goals: list[str]) -> PromptBlock:
        """Build the PPF goals block (required, never trimmed)."""
        if not goals:
            content = "**Goal facts:** (none specified)"
        else:
            items = "\n".join(f"- {g}" for g in goals)
            content = f"**Goal facts:**\n{items}"
        return PromptBlock(
            name="ppf_goals",
            role="user",
            priority=7,
            content=content,
        )

    @staticmethod
    def build_ppf_plan_skeleton_block(plan: list[str]) -> PromptBlock:
        """Build the PPF exploration skeleton block (required, never trimmed)."""
        if not plan:
            content = "**Exploration skeleton:** (none — you design the plan)"
        else:
            items = "\n".join(f"- {b}" for b in plan)
            content = f"**Exploration skeleton (branches, not a step list):**\n{items}"
        return PromptBlock(
            name="ppf_plan_skeleton",
            role="user",
            priority=8,
            content=content,
        )

    @staticmethod
    def build_current_round_actual_block(
        round_idx: int,
        sat_facts: list[str] | None = None,
        unsat_facts: list[str] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> PromptBlock:
        """Build the current round's actual-outcome block.

        Required — never trimmed (priority < 10). This is the framework's
        own fact layer (``evaluate``'s ``goal_status``), expanded in full
        for audit, as distinct from the compressed historical trace (see
        ``build_rounds_trace_block``). Per meta-plan-prompt-and-ppf.md §4.1
        this and the plan gate are the only two hard-retained blocks named
        in the design — everything else is weighted/trimmable, and mixing
        this content into the historical trace would make the one thing
        required to safely audit-before-replan silently droppable under
        budget pressure.
        """
        lines = [
            f"**Round {round_idx} — actual (framework fact layer; audit this "
            "before replanning):**"
        ]
        lines.append(f"- sat: {sat_facts if sat_facts else '(none yet)'}")
        lines.append(f"- unsat: {unsat_facts if unsat_facts else '(none)'}")
        if artifacts:
            lines.append("- artifacts:")
            for a in artifacts:
                aid = a.get("artifact_id", "?")
                summ = a.get("summary", "")
                lines.append(f"  - {aid}: {summ}")
        else:
            lines.append("- artifacts: (none produced this round)")
        return PromptBlock(
            name="current_round_actual",
            role="user",
            priority=9,
            content="\n".join(lines),
        )

    @staticmethod
    def build_round_context_block(
        round_idx: int,
        plan_text: str | None = None,
    ) -> PromptBlock:
        """Build the current round's plan-echo block (required, never trimmed).

        Injected in echo voice per meta-plan-prompt-and-ppf.md §4.3 — this
        is presented as "the plan you wrote," not as a framework directive,
        so the agent revises its own draft rather than executing the
        framework's words.
        """
        lines = [
            f"**Current round:** {round_idx + 1}",
            "",
            "Two acts are required this round: (1) build/run any new steps "
            "via create_step + run_custom_pipeline (skip if nothing new is "
            "needed), then (2) submit_plan (continue) or stop (finish). A "
            "verbatim resubmission keeps the current plan and still counts.",
        ]
        if plan_text is not None:
            lines += [
                "",
                "## This is the plan you wrote last round — revise it, or "
                "resubmit it verbatim to keep it",
                plan_text,
            ]
        return PromptBlock(
            name="round_context",
            role="user",
            priority=9,
            content="\n".join(lines),
        )

    # ----------------------------------------------------- Weighted builders
    # Trimmable — priority >= 10. These are useful but the design doc
    # explicitly puts them in the tunable tier (hints / historical
    # summaries / recall), not the guaranteed-present tier.

    @staticmethod
    def build_ppf_constraints_block(constraints: list[str]) -> PromptBlock:
        """Build the PPF constraints block (weighted, trimmable)."""
        if not constraints:
            content = "**Constraints:** (none)"
        else:
            items = "\n".join(f"- {c}" for c in constraints)
            content = f"**Constraints:**\n{items}"
        return PromptBlock(
            name="ppf_constraints",
            role="user",
            priority=16,
            content=content,
        )

    @staticmethod
    def build_rounds_trace_block(
        rounds: list[dict[str, Any]],
    ) -> PromptBlock:
        """Build the *historical* rounds trace block (weighted, trimmable).

        Prior rounds only — never the current one. The current round's
        actual outcome is the separate, required
        ``build_current_round_actual_block``; mixing them here would make
        the one truly required piece of context silently trimmable.

        Each entry compresses to ``sequence — short summary — outcome``
        per meta-plan-prompt-and-ppf.md §4.4 (full round content remains
        recallable elsewhere; this is not it).

        Args:
            rounds: List of dicts with keys ``round_idx``, ``summary``,
                ``sat_facts``, ``unsat_facts``. Do not include the current,
                in-progress round here.
        """
        if not rounds:
            content = "**Prior rounds:** (none — this is round 1)"
        else:
            lines = ["**Prior rounds (compressed — full text recallable):**"]
            for r in rounds:
                idx = r.get("round_idx", "?")
                summary = r.get("summary", "")
                sat = r.get("sat_facts", [])
                unsat = r.get("unsat_facts", [])
                lines.append(f"- Round {idx} — {summary} — sat={sat}, unsat={unsat}")
            content = "\n".join(lines)
        return PromptBlock(
            name="rounds_trace",
            role="user",
            priority=20,
            content=content,
        )


# ============================================================================
# Singleton instance for backward compatibility
# ============================================================================

_meta_prompt_manager = MetaPlannerPromptManager()

# The assembled prompt without budget trimming — drop-in replacement for
# the old META_PLANNER_SYSTEM_PROMPT constant.
META_PLANNER_SYSTEM_PROMPT = _meta_prompt_manager.assemble()