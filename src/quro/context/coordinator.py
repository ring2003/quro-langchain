"""PromptCoordinator — collects prompt blocks from hooks, orders, assembles.

Design (prompt-management-scaffolding.md §4): prompt blocks are provided by
**StepHooks**; a coordinator owns the hook chain and the cognitive ordering.  A
hook composes ``IPromptBlockProvider`` (``get_blocks``) and returns blocks —
never a pre-joined string.  The framework supplies default hooks (system role /
objective / skills / state-derived blocks); the domain supplies content hooks.

The coordinator owns two independent orders: it runs the hook chain in
registration order, and it assembles blocks in cognitive order (the domain's
``@phase_blocks`` declaration, falling back to ``KIND_ORDER``).
"""

from __future__ import annotations

import json
from typing import Any

from quro.context.blocks import KIND_ORDER, Block, block_kind, registered_blocks, render_block
from quro.core.resources.acl import IAclEngine, Principal
from quro.core.resources.grants import IProjectionGrant
from quro.core.resources.refs import ResourceRef
from quro.core.resources.resolver import IResourceResolver
from quro.steps.hooks import StepHook

# Lazy import to avoid circular dependency at module load time.
_NextInstruction = None


def _get_next_instruction_cls():
    global _NextInstruction
    if _NextInstruction is None:
        from quro.runtime.steering import NextInstruction as _NI
        _NextInstruction = _NI
    return _NextInstruction


class UnknownBlockError(RuntimeError):
    """Raised when a projection grant names a block that is not registered."""


class SystemBlocksHook(StepHook):
    """Framework default: system role identity (``who_you_are`` only).

    Adapter-provided operational context (``project_scope``, ``skills_block``)
    is injected via separate user-role hooks — it must NOT pollute the system
    prompt (adapter-prompt-view-violation.md §4.1).
    """

    name = "system_blocks"
    block_role = "system"

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def get_blocks(self, principal: str) -> dict[str, str]:
        if self._system_prompt:
            return {"who_you_are": self._system_prompt}
        return {}

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


class ProjectScopeBlockHook(StepHook):
    """Framework default: the project scope as a **user** block.

    ``project_scope`` is operational context ("where am I working?"), not
    identity.  It is provided by the adapter, not the domain, so it belongs
    in the user container — keeping the system prompt clean for domain
    identity only (adapter-prompt-view-violation.md §4.2).

    Never trimmed: it is the minimum context a step needs to function.
    """

    name = "project_scope"
    block_role = "user"

    def __init__(self, project_scope: str = "") -> None:
        self._project_scope = project_scope

    def get_blocks(self, principal: str) -> dict[str, str]:
        if self._project_scope:
            return {"project_scope": self._project_scope}
        return {}

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


class ObjectiveBlockHook(StepHook):
    """Framework default: the focused objective (last user block).

    ``objective`` is the LIVE objective the executor must act on.  When a
    steering loop narrows the step's objective per round, the round
    objective is passed here and the original seed is demoted to an
    ``anchor`` (context-only note), so the executor reads a single live
    task instead of the frozen seed (F3).
    """

    name = "objective"
    block_role = "user"

    def __init__(self, step_id: str, objective: str, anchor: str = "") -> None:
        self._step_id = step_id
        self._objective = objective
        self._anchor = anchor

    def get_blocks(self, principal: str) -> dict[str, str]:
        if not self._objective:
            return {}
        content = f"## OBJECTIVE\n{self._objective}"
        if self._anchor:
            content += (
                f"\n\nOriginal step objective (context anchor — do not act on "
                f"this; the OBJECTIVE above is the live round scope): {self._anchor}"
            )
        if self._step_id:
            content += (
                f"\n\nWhen finished, call complete_step(step_id='{self._step_id}')."
            )
        return {"objective": content}

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


class NextInstructionBlockHook(StepHook):
    """Framework default: the next-step instruction block.

    Accepts either a ``NextInstruction`` (structured) or a plain string
    (legacy).  ``kind=31`` so it sorts after ``objective`` (30) — the
    last, most-focused thing the agent reads (primitive-step.md §9.2).

    When a ``NextInstruction`` is provided, the block renders structured
    fields (FOCUS / SCOPE_IN / SCOPE_OUT / RECALL / DONE_WHEN) instead
    of a flat string.
    """

    name = "next_instruction"
    block_role = "user"

    def __init__(
        self,
        instruction: str | Any,
        *,
        state: dict[str, Any] | None = None,
    ) -> None:
        self._instruction = instruction
        self._state = state

    def get_blocks(self, principal: str) -> dict[str, str]:
        if not self._instruction:
            return {}
        NI = _get_next_instruction_cls()
        if isinstance(self._instruction, NI):
            recall = self._resolve_recall(self._instruction)
            return {
                "next_instruction": _render_next_instruction(
                    self._instruction, recall=recall
                ),
            }
        # Legacy str path.
        return {"next_instruction": f"## NEXT INSTRUCTION\n{self._instruction}"}

    def _resolve_recall(self, ni: Any) -> list[str] | None:
        """Resolve RECALL ids against the carried ArtifactIndex.

        Each recalled artifact id is rewritten to ``id:<summary>`` so the
        executor addresses the artifact by its id + a short human label
        instead of an opaque number (Anti-Pattern H/I). Unknown ids pass
        through unchanged.  Returns ``None`` when no index is available so the
        caller falls back to the plain id list.
        """
        if self._state is None:
            return None
        try:
            from quro.context.artifact_index import build_artifact_index
            from quro.core.resources.acl import acl_for_state, principal_from_state

            view = build_artifact_index(
                self._state, principal_from_state(self._state), acl_for_state(self._state)
            )
        except Exception:  # noqa: BLE001 - index is a best-effort hint
            return None
        if not view.artifact_count:
            return None
        lookup = {e.id: e.summary for e in view.artifacts}
        resolved: list[str] = []
        for aid in (ni.recall or []):
            summary = lookup.get(aid)
            resolved.append(f"{aid}:{summary}" if summary else aid)
        return resolved or None

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


def _render_next_instruction(ni: Any, *, recall: list[str] | None = None) -> str:
    """Render a ``NextInstruction`` into the ``NEXT INSTRUCTION`` block text.

    *recall*, when provided, overrides ``ni.recall`` — the caller supplies the
    ids resolved against the ArtifactIndexView (rewritten to
    ``id:<summary>``), so the executor addresses recall artifacts by id rather
    than an opaque number.
    """
    lines = [f"## NEXT INSTRUCTION (round {ni.round_index})"]
    lines.append(f"FOCUS: {ni.focus}")
    if ni.scope_in:
        lines.append(f"LOOK AT: {', '.join(ni.scope_in)}")
    if ni.scope_out:
        lines.append(f"DO NOT RE-EXAMINE: {', '.join(ni.scope_out)}")
    resolved_recall = recall if recall is not None else ni.recall
    if resolved_recall:
        lines.append(f"RECALL IF NEEDED: {', '.join(resolved_recall)}")
    if ni.done_when:
        lines.append(f"THIS ROUND IS DONE WHEN: {ni.done_when}")
    if ni.is_final_round:
        lines.append(
            "This is the final round of this step — when the round-done "
            "condition is met, produce a conclusive result."
        )
    return "\n".join(lines)


class SkillsBlockHook(StepHook):
    """Framework default: the skill block (Phase B declared/undeclared skills)."""

    name = "skills"
    block_role = "user"

    def __init__(self, skills_block: str = "") -> None:
        self._skills_block = skills_block

    def get_blocks(self, principal: str) -> dict[str, str]:
        return {"skills": self._skills_block} if self._skills_block else {}

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


class StateBlocksHook(StepHook):
    """Framework default: render state-derived user blocks by declared name."""

    name = "state_blocks"
    block_role = "user"

    def __init__(
        self,
        block_names: list[str],
        state: dict[str, Any],
        phase: str,
        budget: int,
        principal: str | None,
    ) -> None:
        self._block_names = block_names
        self._state = state
        self._phase = phase
        self._budget = budget
        self._principal = principal

    def get_blocks(self, principal: str) -> dict[str, str]:
        blocks: dict[str, str] = {}
        renderable = set(registered_blocks())
        for name in self._block_names:
            if name not in renderable:
                continue  # framework default block (objective/skills) — provided elsewhere
            content = render_block(
                name, self._state,
                phase=self._phase, budget=self._budget, principal=self._principal,
            )
            if content:
                blocks[name] = content
        return blocks

    def on_pre_step(self, step, context):  # noqa: D401
        return None

    def on_post_step(self, step, result, context):  # noqa: D401
        return None


class PromptCoordinator:
    """Collect blocks from hooks, order them, assemble system/user containers."""

    def __init__(
        self,
        hooks: list[Any],
        *,
        principal: str,
        domain: Any = None,
        phase: str = "",
        grants: IProjectionGrant | None = None,
        acl: IAclEngine | None = None,
        acl_principal: Principal | None = None,
        resolver: IResourceResolver | None = None,
    ) -> None:
        self._hooks = hooks
        self._principal = principal
        self._domain = domain
        self._phase = phase
        self._grants = grants
        self._acl = acl
        self._acl_principal = acl_principal
        self._resolver = resolver

    def assemble(self, *, budget: int = 0) -> tuple[dict[str, str], dict[str, str]]:
        """Return ``(system_blocks, user_blocks)`` in cognitive order.

        Order is ``KIND_ORDER`` (framework constant).  When *budget* is set,
        blocks are trimmed by kind priority (lowest-priority kind first) —
        the objective and next_instruction blocks are always kept last.
        Projection grants (phase 1) are applied after collection: a matching
        grant whose ``source`` passes the ACL check renders its declared block
        from the source content.
        """
        ordered = self._order(self._collect())
        # Projected blocks are appended, then re-ordered so the "objective
        # last" invariant survives a grant adding a block.
        ordered = self._order(self._apply_grants(ordered))
        if budget > 0:
            ordered = self._trim(ordered, budget)
        system_blocks: dict[str, str] = {}
        user_blocks: dict[str, str] = {}
        for block in ordered:
            target = system_blocks if block.role == "system" else user_blocks
            if block.name not in target:
                target[block.name] = block.content
        return system_blocks, user_blocks

    def _trim(self, blocks: list[Block], budget: int) -> list[Block]:
        """Drop the lowest-priority blocks until content fits *budget*.

        Blocks are already in cognitive order (``KIND_ORDER``); trim drops
        from the highest kind rank (least critical) first.  ``objective``,
        ``next_instruction``, ``project_scope``, and ``who_you_are`` are never
        dropped — they are the minimum context a step needs to function
        (adapter-prompt-view-violation.md §4.2).
        """
        total = sum(len(b.content) for b in blocks)
        if total <= budget:
            return blocks
        # Walk from least-critical to most-critical, dropping until fit.
        _NEVER_TRIM = frozenset({
            "objective", "who_you_are", "next_instruction", "project_scope",
        })
        for idx in range(len(blocks) - 1, -1, -1):
            if blocks[idx].name in _NEVER_TRIM:
                continue
            total -= len(blocks[idx].content)
            blocks = blocks[:idx] + blocks[idx + 1:]
            if total <= budget:
                break
        return blocks

    # -- internals ---------------------------------------------------------

    def _collect(self) -> list[Block]:
        blocks: list[Block] = []
        for hook in self._hooks:
            get_blocks = getattr(hook, "get_blocks", None)
            if get_blocks is None:
                continue
            role = getattr(hook, "block_role", "user")
            for name, content in get_blocks(self._principal).items():
                if content:
                    blocks.append(
                        Block(
                            name=name,
                            kind=block_kind(name),
                            content=content,
                            role=role,
                        )
                    )
        return blocks

    def _order(self, blocks: list[Block]) -> list[Block]:
        # ``KIND_ORDER`` is a framework constant the domain cannot override:
        # identity → duties → constraints → … → objective last.  The domain's
        # ``@phase_blocks`` declaration is a *participation set* only (which
        # blocks compose a phase) — it is NOT an ordering authority.
        def key(block: Block) -> tuple[int, int, str]:
            role_rank = 0 if block.role == "system" else 1
            kind_rank = KIND_ORDER.get(block.kind, 40)
            return (role_rank, kind_rank, block.name)

        return sorted(blocks, key=key)

    def _declared_blocks(self) -> set[str]:
        """The participation set for the current phase (``@phase_blocks``)."""
        mapping = getattr(self._domain, "PHASE_BLOCKS", None) if self._domain else None
        if not mapping:
            return set()
        return set(mapping.get(self._phase, []))

    # -- projection grants (architecture §6) -------------------------------

    def _apply_grants(self, blocks: list[Block]) -> list[Block]:
        """Apply matching projection grants on top of the collected blocks.

        Two doors, neither removed (design doc §9.4): the grant's ``source``
        must pass the ACL resource check first — a denied source contributes
        nothing.  A block the default projection already rendered is left
        untouched.  Unknown block names fail fast at assembly.
        """
        if self._grants is None or self._acl is None or self._resolver is None:
            return blocks
        principal = self._acl_principal or Principal(step_id=self._principal)
        existing = {b.name for b in blocks}
        renderable = set(registered_blocks())

        for grant in self._grants.grants_for(self._principal, self._phase):
            # Reference integrity fails fast BEFORE the ACL gate: an unknown
            # block name is an assembly error regardless of the source verdict.
            unknown = [b for b in grant.blocks if b not in renderable]
            if unknown:
                raise UnknownBlockError(
                    f"projection grant for {grant.principal!r} names "
                    f"unknown block {unknown[0]!r} — registered blocks: "
                    f"{sorted(renderable)}"
                )
            verdict = self._acl.check(principal, grant.source, "read")
            if not verdict.allowed:
                continue  # denied source → contributes nothing
            for block_name in grant.blocks:
                if block_name in existing:
                    continue  # default projection already present
                content = self._render_source(grant.source)
                if content:
                    blocks.append(Block(
                        name=block_name,
                        kind=block_kind(block_name),
                        content=content,
                        role="user",
                    ))
                    existing.add(block_name)
        return blocks

    def _render_source(self, source: ResourceRef) -> str:
        """Render a grant ``source`` resource into block text."""
        data = self._resolver.read(source)
        return _render_source_content(data)


def _render_source_content(data: Any) -> str:
    """Render resolver output into a compact text block."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "summary" in data or "body" in data:
            lines = []
            aid = data.get("artifact_id", "")
            summary = str(data.get("summary", "") or "").strip()
            if aid or summary:
                lines.append(f"- {aid}: {summary}".rstrip(": "))
            body = str(data.get("body", "") or "").strip()
            if body:
                lines.append(body)
            return "\n".join(lines)
        return json.dumps(data, ensure_ascii=False, indent=2)
    if isinstance(data, (list, tuple)):
        return "\n".join(str(x) for x in data)
    return str(data)
