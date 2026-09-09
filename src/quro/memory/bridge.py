"""Memory bridges — adapters for persistent memory backends (Phase 2).

Implements ``IMemoryBridge`` for:

- ``QuroMemoryBridge`` — adapts quro-memory's ``add_memory`` / ``recall`` API
- ``NoopMemoryBridge`` — null adapter, discards all writes (Phase 1 default)

QuroMemoryBridge maintains an in-process write cache to bridge the async
distillation gap (blueprint §9.4).
"""

from __future__ import annotations

from typing import Any

from quro.algorithm.signature import ProblemSignature
from quro.memory.protocols import IMemoryBridge


# ============================================================================
# NoopMemoryBridge — null adapter
# ============================================================================


class NoopMemoryBridge(IMemoryBridge):
    """Null bridge — discards all writes, returns empty recalls.

    Safe default for Phase 1 when quro-memory is not yet wired.
    """

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
        return None

    def recall(
        self,
        signature: ProblemSignature,
        *,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return []

    def recall_patterns(
        self,
        pattern_tags: list[str],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return []

    def flush(self) -> None:
        pass


# ============================================================================
# QuroMemoryBridge — quro-memory adapter
# ============================================================================


class QuroMemoryBridge(IMemoryBridge):
    """Adapter that bridges quro-memory's ``add_memory`` / ``recall`` API.

    Uses in-process ``import quro_memory`` (blueprint §8 in-process mode) for
    zero MCP overhead.  Maintains an in-process write cache so ``recall``
    immediately after ``distill`` returns the just-written entry even if the
    quro-memory background worker hasn't indexed it yet (blueprint §9.4).

    Args:
        api: An optional pre-configured quro-memory API instance.  If None,
            the bridge will ``import quro_memory`` and use the module-level
            ``add_memory`` / ``recall`` functions.

    Usage::

        bridge = QuroMemoryBridge()
        bridge.distill(sig, 0, body="...", tags=["problem:abc123"])
        results = bridge.recall(sig)
    """

    def __init__(self, api: Any = None) -> None:
        self._api = api
        self._cache: dict[str, dict[str, Any]] = {}  # memory_id → entry

    # ----------------------------------------------------------------- api
    def _get_api(self) -> Any:
        """Lazy-import quro_memory if no pre-configured api."""
        if self._api is not None:
            return self._api
        try:
            import quro_memory.api as qm_api

            self._api = qm_api
            return qm_api
        except ImportError:
            return None

    # -------------------------------------------------------------- distill

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
        """Write a round's compacted record to quro-memory.

        Uses ``add_memory`` with an idempotent memory_id so repeated writes
        of the same round are safe.
        """
        api = self._get_api()

        memory_id = signature.to_memory_id(round_index)
        resolved_tags = tags or signature.to_memory_tags()

        # Always write to in-process cache (bridges async distillation gap).
        entry: dict[str, Any] = {
            "id": memory_id,
            "body": body,
            "tags": resolved_tags,
            "type": memory_type,
            "importance": importance,
        }
        self._cache[memory_id] = entry

        if api is None:
            return memory_id  # cache-only mode

        try:
            result = api.add_memory(
                body=body,
                memory_type=memory_type,
                tags=resolved_tags,
                importance=importance,
                memory_id=memory_id,
            )
            return result.get("id") if isinstance(result, dict) else memory_id
        except Exception:
            return memory_id

    # --------------------------------------------------------------- recall

    def recall(
        self,
        signature: ProblemSignature,
        *,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall historical records for *signature*.

        Merges in-process cache with quro-memory results.
        """
        api = self._get_api()

        backend_results: list[dict[str, Any]] = []
        if api is not None:
            try:
                raw = api.recall(
                    query=signature.sig,
                    filters={"type": memory_types} if memory_types else None,
                )
                if isinstance(raw, list):
                    backend_results = raw[:limit]
            except Exception:
                pass

        # Merge cache entries that match the signature.
        cache_matches: list[dict[str, Any]] = []
        sig_prefix = signature.sig
        for mid, entry in self._cache.items():
            if mid.startswith(sig_prefix):
                if memory_types is None or entry.get("type") in memory_types:
                    cache_matches.append(entry)
                    if len(cache_matches) + len(backend_results) >= limit:
                        break

        # Cache entries first (newest), then backend.
        merged = cache_matches + backend_results
        return merged[:limit]

    def recall_patterns(
        self,
        pattern_tags: list[str],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cross-project recall by pattern tags (blueprint §6 L2)."""
        api = self._get_api()
        results: list[dict[str, Any]] = []

        if api is not None:
            for tag in pattern_tags:
                try:
                    raw = api.recall(query=tag)
                    if isinstance(raw, list):
                        results.extend(raw)
                except Exception:
                    pass

        # Add cache matches.
        for entry in self._cache.values():
            if any(tag in entry.get("tags", []) for tag in pattern_tags):
                results.append(entry)

        return results[:limit]

    # ---------------------------------------------------------------- flush

    def flush(self) -> None:
        """Sync in-process cache (no-op for quro-memory — cache is
        live-updated on write)."""
        pass
