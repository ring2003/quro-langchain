"""Memory protocols — abstractions for working memory and cross-project recall.

All consumers (``MetaPlannerLoop``, ``Session``) depend on
these protocols, **not** on concrete implementations.  This allows:

- Phase 1: ``InMemoryWorkingMemory`` (no external deps)
- Phase 2: ``QuroMemoryBridge`` (quro-memory adapter)
- Future: any other backend (vector DB, SQLite, Redis, …)

Protocols follow the blueprint §6 layered design:

::

    L0  ``IWorkingMemory``     — in-session state injection
    L1+ ``IMemoryBridge``      — per-round distillation + cross-project recall
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quro.algorithm.signature import ProblemSignature


# ============================================================================
# IWorkingMemory — L0 in-session state (blueprint §6)
# ============================================================================


class IWorkingMemory(ABC):
    """In-session working memory for injecting compacted state into each round.

    Phase 1 (blueprint §10): a simple dict-based accumulator that carries
    UNSAT_facts and progress summaries across rounds within a single
    agent-driven loop run.  Later phases may add compression,
    priority-aware eviction, etc.

    All mutations are in-process and synchronous — no persistence.
    """

    @abstractmethod
    def update(self, round_index: int, summary: dict[str, Any]) -> None:
        """Record a round's compacted summary.

        Args:
            round_index: 0-based round number.
            summary: Dict with keys like ``unsat_facts``, ``sat_facts``,
                ``disposition``, ``cause``, ``completed_steps``.
        """
        ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return the current working memory for injection into the
        planner's next-round context.

        Returns:
            Dict suitable for serialization / prompt injection.
        """
        ...

    @abstractmethod
    def previous_unsat_facts(self) -> list[str] | None:
        """Return UNSAT_facts from the previous round, or None.

        Used by the evaluate stage for stall detection.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        ...

    @abstractmethod
    def get_l1_context(self) -> dict[str, Any]:
        """Return a lightweight context summary for L1 MetaPlanner injection.

        This is a **capability interface** — the current implementation
        uses rule-based extraction from accumulated rounds.  Future
        phases may replace this with LLM-based distillation.

        Returns:
            A dict with keys like ``rounds``, ``latest_unsat_facts``,
            ``total_rounds``, ``disposition_history``.
        """
        ...


# ============================================================================
# IMemoryBridge — L1/L2 persistent memory (blueprint §6, §8)
# ============================================================================


class IMemoryBridge(ABC):
    """Bridge to a persistent memory backend for per-round distillation (L1)
    and cross-project recall (L2).

    Implementations include:
    - ``QuroMemoryBridge`` — adapts quro-memory's ``add_memory`` / ``recall``
    - ``NoopMemoryBridge``  — discards all writes, returns empty recalls

    Async caveat (blueprint §9.4): some backends index asynchronously; a
    ``recall`` immediately after ``distill`` may miss the just-written entry.
    Implementations SHOULD maintain an in-process write cache to bridge this gap.
    """

    @abstractmethod
    def distill(
        self,
        signature: ProblemSignature,
        round_index: int,
        *,
        body: str,
        tags: list[str] | None = None,
        memory_type: str = "fact",
        importance: float = 0.7,
    ) -> str | None:
        """Persist a round's compacted record.

        Args:
            signature: Stable problem identity.
            round_index: 0-based round number.
            body: Text body for the memory entry.
            tags: Tags for filtering / recall.
            memory_type: quro-memory entry type (``"fact"``,
                ``"episode"``, ``"procedural"``).
            importance: 0.0–1.0 importance score.

        Returns:
            The memory entry id, or None if the backend is unavailable.
        """
        ...

    @abstractmethod
    def recall(
        self,
        signature: ProblemSignature,
        *,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall historical records for *signature*.

        Args:
            signature: Problem identity to recall.
            memory_types: Filter by memory type; None = all.
            limit: Maximum entries to return.

        Returns:
            List of memory entry dicts (keys: ``id``, ``body``, ``tags``,
            ``type``, ``importance``, …), newest first.
        """
        ...

    @abstractmethod
    def recall_patterns(
        self,
        pattern_tags: list[str],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cross-project recall by pattern tags (blueprint §6 L2).

        Args:
            pattern_tags: Tags like ``"pattern:goal_unmet"``,
                ``"pattern:htn_unsolvable"``.
            limit: Maximum entries.

        Returns:
            Matching entries from any project.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Ensure all pending writes are visible to subsequent ``recall`` calls.

        Implementations that use an in-process write cache should sync it here.
        """
        ...
