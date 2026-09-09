"""Context Layer data models.

Pure-Python dataclasses and value objects for the context management
orchestration layer.  Zero dependency on quro internal modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Block kind literal
# ---------------------------------------------------------------------------

BlockKind = str  # "system" | "user" | "conversation" | "tool"


# ---------------------------------------------------------------------------
# ContextBlock — tree node
# ---------------------------------------------------------------------------


@dataclass
class ContextBlock:
    """A node in the context block tree.

    Blocks are organised hierarchically: top-level blocks are keyed by
    ``kind`` (system / user / conversation), and each may hold nested
    child blocks (e.g. ``system.constraints``).
    """

    id: str
    """Unique identifier, e.g. ``"system.constraints"``."""

    name: str
    """Human-readable name, e.g. ``"constraints"``."""

    kind: BlockKind
    """Top-level category: ``"system"``, ``"user"``, ``"conversation"``, or ``"tool"``."""

    content: str = ""
    """Current text content of this block."""

    estimated_tokens: int = 0
    """Estimated token count, updated on each :meth:`ISessionContext.assemble`."""

    actual_tokens: int = 0
    """Actual token count back-filled from LLM ``usage_metadata`` on
    :meth:`ISessionContext.update`."""

    children: list[ContextBlock] = field(default_factory=list)
    """Nested child blocks."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata, e.g.
    ``{"source": "IWorkingMemory.snapshot", "compressible": True, "priority": "required"}``.
    """

    @property
    def total_estimated_tokens(self) -> int:
        """Recursive sum of estimated tokens for this block and its children."""
        return self.estimated_tokens + sum(
            c.total_estimated_tokens for c in self.children
        )

    @property
    def total_actual_tokens(self) -> int:
        """Recursive sum of actual tokens for this block and its children."""
        return self.actual_tokens + sum(
            c.total_actual_tokens for c in self.children
        )


# ---------------------------------------------------------------------------
# BlockTree — hierarchical block container
# ---------------------------------------------------------------------------


class BlockTree:
    """Mutable tree of :class:`ContextBlock` nodes.

    Provides path-based CRUD (e.g. ``"system.constraints"``) and a
    strategy-driven trim operation.
    """

    def __init__(self) -> None:
        self._roots: dict[str, ContextBlock] = {}

    # -- write -----------------------------------------------------------

    def add_block(
        self,
        path: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContextBlock:
        """Add or update a block at *path*.

        Intermediate nodes are created automatically.  A path such as
        ``"system.constraints"`` creates (or reuses) a root ``"system"``
        block and then a ``"constraints"`` child inside it.

        Args:
            path: Dot-separated block path.
            content: Text content for the leaf block.
            metadata: Optional metadata dict.

        Returns:
            The created or updated leaf block.
        """
        parts = path.split(".")
        root_kind = parts[0]
        if root_kind not in self._roots:
            self._roots[root_kind] = ContextBlock(
                id=root_kind, name=root_kind, kind=root_kind
            )

        current = self._roots[root_kind]
        for i, part in enumerate(parts[1:], start=1):
            full_id = ".".join(parts[: i + 1])
            # look for existing child with this id
            child = next((c for c in current.children if c.id == full_id), None)
            if child is None:
                child = ContextBlock(
                    id=full_id, name=part, kind=root_kind
                )
                current.children.append(child)
            current = child

        current.content = content
        if metadata is not None:
            current.metadata.update(metadata)
        return current

    # -- read ------------------------------------------------------------

    def get_block(self, path: str) -> ContextBlock | None:
        """Return the block at *path*, or ``None``.

        Args:
            path: Dot-separated block path.
        """
        parts = path.split(".")
        root = self._roots.get(parts[0])
        if root is None:
            return None
        current = root
        for part in parts[1:]:
            full_id = ".".join([current.id, part]) if current.id else part
            child = next((c for c in current.children if c.id == full_id), None)
            if child is None:
                return None
            current = child
        return current

    def remove_block(self, path: str) -> None:
        """Remove a block and its entire subtree.

        Args:
            path: Dot-separated block path.  Removing a root removes all
                blocks of that kind.
        """
        parts = path.split(".")
        if len(parts) == 1:
            self._roots.pop(parts[0], None)
            return

        parent = self.get_block(".".join(parts[:-1]))
        if parent is None:
            return
        target_id = path
        parent.children = [c for c in parent.children if c.id != target_id]

    # -- traversal -------------------------------------------------------

    def walk(self) -> Iterator[tuple[str, ContextBlock]]:
        """Depth-first walk yielding ``(path, block)`` for every node."""
        for root in self._roots.values():
            yield from self._walk_node(root)

    def _walk_node(self, node: ContextBlock) -> Iterator[tuple[str, ContextBlock]]:
        yield node.id, node
        for child in node.children:
            yield from self._walk_node(child)

    # -- trim ------------------------------------------------------------

    def trim(self, strategy: dict[str, TrimAction]) -> list[str]:
        """Apply a trim strategy, returning paths of modified blocks.

        Args:
            strategy: Mapping of block path patterns to :class:`TrimAction`.
                Example::

                    {"user.hints": TrimAction.COMPRESS(max_chars=500),
                     "conversation.*": TrimAction.DROP_OLDEST(keep_last=10)}

        Returns:
            Paths of blocks that were modified.
        """
        modified: list[str] = []
        for path, action in strategy.items():
            if path.endswith(".*"):
                # wildcard — apply to all children of the prefix
                prefix = path[:-2]
                parent = self.get_block(prefix) if prefix else None
                if parent is not None:
                    for child in parent.children:
                        if self._apply_action(child, action):
                            modified.append(child.id)
            else:
                block = self.get_block(path)
                if block is not None and self._apply_action(block, action):
                    modified.append(block.id)
        return modified

    @staticmethod
    def _apply_action(block: ContextBlock, action: TrimAction) -> bool:
        if action.kind == "compress":
            if len(block.content) > action.max_chars:
                block.content = block.content[: action.max_chars]
                block.metadata["trimmed"] = True
                return True
        elif action.kind == "drop_oldest":
            # Handled at the message-composer level — mark for later processing.
            block.metadata["trim_drop_oldest"] = True
            block.metadata["trim_keep_last"] = action.keep_last
            return True
        elif action.kind == "drop":
            block.content = ""
            block.metadata["trimmed"] = True
            return True
        return False


# ---------------------------------------------------------------------------
# TrimAction
# ---------------------------------------------------------------------------


@dataclass
class TrimAction:
    """Description of a single trim operation on a block.

    Create via factory class methods rather than the constructor directly.
    """

    kind: str
    """``"compress"``, ``"drop_oldest"``, or ``"drop"``."""

    max_chars: int = 0
    """Maximum characters for ``compress``."""

    keep_last: int = 0
    """Number of items to keep for ``drop_oldest``."""

    @classmethod
    def COMPRESS(cls, max_chars: int) -> TrimAction:
        """Truncate block content to *max_chars* characters."""
        return cls(kind="compress", max_chars=max_chars)

    @classmethod
    def DROP_OLDEST(cls, keep_last: int) -> TrimAction:
        """Keep only the most recent *keep_last* conversation rounds."""
        return cls(kind="drop_oldest", keep_last=keep_last)

    @classmethod
    def DROP(cls) -> TrimAction:
        """Remove the block entirely."""
        return cls(kind="drop")


# ---------------------------------------------------------------------------
# Token ledger
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    """A single row in the token ledger — one block × one round."""

    path: str
    """Block path, e.g. ``"system.constraints"``."""

    round_index: int
    """Which conversation round this entry belongs to."""

    estimated_tokens: int = 0
    """Token count estimated during :meth:`ISessionContext.assemble`."""

    actual_prompt_tokens: int = 0
    """Prompt tokens attributed to this block after LLM response."""

    actual_completion_tokens: int = 0
    """Completion tokens attributed to this block (usually 0 except for
    the virtual ``"completion"`` block)."""


@dataclass
class BlockTokenReport:
    """Per-block token summary."""

    path: str
    estimated_tokens: int = 0
    actual_prompt_tokens: int = 0
    percentage_of_total: float = 0.0
    rounds_contributed: int = 0


@dataclass
class RoundTokenReport:
    """Per-round token summary."""

    round_index: int
    estimated_tokens: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0


@dataclass
class LedgerSummary:
    """Aggregated token ledger summary."""

    total_estimated_tokens: int = 0
    total_actual_prompt_tokens: int = 0
    total_actual_completion_tokens: int = 0
    per_block: dict[str, BlockTokenReport] = field(default_factory=dict)
    per_round: dict[int, RoundTokenReport] = field(default_factory=dict)


class TokenLedger:
    """Per-block, per-round token accounting.

    Two-phase: *estimate* during message assembly, then *actual* after
    the LLM response arrives (allocated proportionally across blocks).
    """

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    # -- recording -------------------------------------------------------

    def record_estimate(self, path: str, round_index: int, tokens: int) -> None:
        """Record the estimated token count for *path* at *round_index*.

        Called from :meth:`ISessionContext.assemble`.
        """
        self.entries.append(
            LedgerEntry(path=path, round_index=round_index, estimated_tokens=tokens)
        )

    def record_actual(
        self,
        round_index: int,
        input_tokens: int,
        output_tokens: int,
        *,
        allocation: dict[str, float] | None = None,
    ) -> None:
        """Back-fill actual token counts after LLM response.

        *input_tokens* are distributed across blocks according to
        *allocation* (a mapping of block path → proportion, e.g.
        ``{"role": 0.12, "problem": 0.18}``).  If
        *allocation* is ``None``, input tokens are spread evenly across
        the entries for this round.

        *output_tokens* are credited to a virtual ``"completion"`` block.

        Args:
            round_index: The round being updated.
            input_tokens: Total prompt tokens from ``usage_metadata``.
            output_tokens: Total completion tokens from ``usage_metadata``.
            allocation: Optional per-block proportion map (values should
                sum to ~1.0).
        """
        round_entries = [e for e in self.entries if e.round_index == round_index]
        if not round_entries:
            return

        if allocation:
            for entry in round_entries:
                proportion = allocation.get(entry.path, 0.0)
                entry.actual_prompt_tokens = int(input_tokens * proportion)
        else:
            share = input_tokens // len(round_entries)
            for entry in round_entries:
                entry.actual_prompt_tokens = share

        # Credit completion tokens to a virtual block
        self.entries.append(
            LedgerEntry(
                path="completion",
                round_index=round_index,
                actual_completion_tokens=output_tokens,
            )
        )

    # -- summarise -------------------------------------------------------

    def summarize(self) -> LedgerSummary:
        """Return an aggregated summary of all ledger entries."""
        summary = LedgerSummary()
        block_rounds: dict[str, set[int]] = {}

        for entry in self.entries:
            summary.total_estimated_tokens += entry.estimated_tokens
            summary.total_actual_prompt_tokens += entry.actual_prompt_tokens
            summary.total_actual_completion_tokens += entry.actual_completion_tokens

            # per-block rollup
            report = summary.per_block.get(entry.path)
            if report is None:
                report = BlockTokenReport(path=entry.path)
                summary.per_block[entry.path] = report
            report.estimated_tokens += entry.estimated_tokens
            report.actual_prompt_tokens += entry.actual_prompt_tokens
            block_rounds.setdefault(entry.path, set()).add(entry.round_index)

            # per-round rollup
            rr = summary.per_round.get(entry.round_index)
            if rr is None:
                rr = RoundTokenReport(round_index=entry.round_index)
                summary.per_round[entry.round_index] = rr
            rr.estimated_tokens += entry.estimated_tokens
            rr.actual_prompt_tokens += entry.actual_prompt_tokens
            rr.actual_completion_tokens += entry.actual_completion_tokens

        # Fill percentages and round counts
        total_actual = summary.total_actual_prompt_tokens or 1  # avoid /0
        for path, report in summary.per_block.items():
            report.percentage_of_total = round(
                report.actual_prompt_tokens / total_actual * 100, 1
            )
            report.rounds_contributed = len(block_rounds.get(path, set()))

        return summary


# ---------------------------------------------------------------------------
# SessionContextConfig
# ---------------------------------------------------------------------------


@dataclass
class SessionContextConfig:
    """Session-level context configuration.

    Passed to :meth:`ISessionContext.open_session`.
    """

    # Block structure
    system_blocks: list[str] = field(default_factory=lambda: [
        "role", "objectives", "constraints", "tools", "hints",
    ])
    """Ordered list of system block names."""

    user_blocks: list[str] = field(default_factory=lambda: [
        "problem", "hints", "objective",
    ])
    """Ordered list of user block names."""

    # Capacity
    max_tokens: int = 128000
    """Total token budget (0 = unlimited)."""

    max_conversation_rounds: int = 20
    """Maximum conversation rounds to retain."""

    # Trim
    trim_strategy: str = "last"
    """Built-in strategy: ``"last"``, ``"first"``, or ``"block_aware"``."""

    compression_threshold: float = 0.8
    """Trigger trimming when estimated tokens exceed this fraction of *max_tokens*."""

    # Reporting
    report_interval_rounds: int = 5
    """Print a token report every N rounds."""

    report_detail: str = "block"
    """Report granularity: ``"block"``, ``"round"``, or ``"full"``."""


# ---------------------------------------------------------------------------
# SessionContextReport
# ---------------------------------------------------------------------------


@dataclass
class SessionContextReport:
    """Final report produced by :meth:`ISessionContext.close_session`."""

    total_rounds: int = 0
    """Number of LLM rounds executed in this session."""

    ledger_summary: LedgerSummary = field(default_factory=LedgerSummary)
    """Aggregated token ledger."""

    trim_events: int = 0
    """How many times trimming was triggered."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary session metadata."""

    def format(self) -> str:
        """Return a human-readable table of the token report."""
        ls = self.ledger_summary
        lines = [
            f"══ Context Token Report ({self.total_rounds} rounds) ═══════════════════",
            f"{'Block':<28} {'Est':>6}  {'Actual':>7}  {'%Total':>7}",
            "─" * 56,
        ]
        # Sort by actual prompt tokens descending
        sorted_blocks = sorted(
            ls.per_block.values(),
            key=lambda r: r.actual_prompt_tokens,
            reverse=True,
        )
        for report in sorted_blocks:
            lines.append(
                f"{report.path:<28} {report.estimated_tokens:>6}  "
                f"{report.actual_prompt_tokens:>7}  {report.percentage_of_total:>6.1f}%"
            )
        lines.append("─" * 56)
        lines.append(
            f"{'TOTAL (prompt)':<28} {ls.total_estimated_tokens:>6}  "
            f"{ls.total_actual_prompt_tokens:>7}"
        )
        lines.append(f"completion tokens: {ls.total_actual_completion_tokens:>10}")
        if self.metadata.get("budget"):
            budget = self.metadata["budget"]
            used = ls.total_actual_prompt_tokens + ls.total_actual_completion_tokens
            pct = used / budget * 100 if budget else 0
            lines.append(
                f"Session budget: {budget:,}  Used: {used:,} ({pct:.1f}%)"
            )
        return "\n".join(lines)
