"""Tests for ``src/quro/memory/`` — protocols and implementations."""

import pytest

from quro.algorithm.signature import ProblemSignature
from quro.memory.bridge import NoopMemoryBridge, QuroMemoryBridge
from quro.memory.protocols import IMemoryBridge, IWorkingMemory
from quro.memory.working import InMemoryWorkingMemory


# ============================================================================
# InMemoryWorkingMemory
# ============================================================================


class TestInMemoryWorkingMemory:
    def test_empty(self) -> None:
        wm = InMemoryWorkingMemory()
        assert wm.previous_unsat_facts() is None
        snap = wm.snapshot()
        assert snap["total_rounds"] == 0

    def test_update_and_snapshot(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.update(0, {
            "unsat_facts": ["a"],
            "sat_facts": ["b"],
            "disposition": "add_steps",
            "cause": "tests fail",
            "completed_steps": 3,
        })
        assert wm.previous_unsat_facts() == ["a"]

        snap = wm.snapshot()
        assert snap["total_rounds"] == 1
        assert snap["latest"]["unsat_facts"] == ["a"]
        assert snap["latest"]["sat_facts"] == ["b"]
        assert snap["latest"]["disposition"] == "add_steps"
        assert snap["latest"]["completed_steps"] == 3

    def test_multiple_rounds(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.update(0, {"unsat_facts": ["a", "b"]})
        wm.update(1, {"unsat_facts": ["a"]})
        # Latest unsat is from round 1.
        assert wm.previous_unsat_facts() == ["a"]
        snap = wm.snapshot()
        assert snap["total_rounds"] == 2
        assert len(snap["rounds"]) == 2

    def test_update_without_unsat_facts(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.update(0, {"disposition": "continue"})
        # previous_unsat_facts stays None (not overridden by missing key).
        assert wm.previous_unsat_facts() is None

    def test_reset(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.update(0, {"unsat_facts": ["a"]})
        wm.reset()
        assert wm.previous_unsat_facts() is None
        assert wm.snapshot()["total_rounds"] == 0

    def test_protocol_compliance(self) -> None:
        """InMemoryWorkingMemory satisfies IWorkingMemory protocol."""
        wm: IWorkingMemory = InMemoryWorkingMemory()
        wm.update(0, {})
        assert isinstance(wm.snapshot(), dict)


# ============================================================================
# NoopMemoryBridge
# ============================================================================


class TestNoopMemoryBridge:
    def _sig(self) -> ProblemSignature:
        return ProblemSignature.compute("test", [])

    def test_distill_returns_none(self) -> None:
        bridge = NoopMemoryBridge()
        result = bridge.distill(self._sig(), 0, body="test")
        assert result is None

    def test_recall_returns_empty(self) -> None:
        bridge = NoopMemoryBridge()
        assert bridge.recall(self._sig()) == []

    def test_recall_patterns_returns_empty(self) -> None:
        bridge = NoopMemoryBridge()
        assert bridge.recall_patterns(["pattern:goal_unmet"]) == []

    def test_flush_noop(self) -> None:
        bridge = NoopMemoryBridge()
        bridge.flush()  # no error

    def test_protocol_compliance(self) -> None:
        bridge: IMemoryBridge = NoopMemoryBridge()
        assert bridge.recall(self._sig()) == []


# ============================================================================
# QuroMemoryBridge
# ============================================================================


class TestQuroMemoryBridge:
    def _sig(self) -> ProblemSignature:
        return ProblemSignature.compute("test problem", ["a"])

    def test_noop_when_quro_memory_not_installed(self) -> None:
        """Without quro-memory, distill returns memory_id (cache-only)."""
        bridge = QuroMemoryBridge()
        # Completely disable the API so we test cache-only mode.
        bridge._get_api = lambda: None  # type: ignore[method-assign]

        sig = self._sig()
        result = bridge.distill(sig, 0, body="test", tags=[])
        assert result is not None  # returns memory_id even in cache-only mode
        # recall should find the cached entry.
        recalled = bridge.recall(sig)
        assert len(recalled) == 1
        assert recalled[0]["body"] == "test"

    def test_in_process_cache_bridges_async_gap(self) -> None:
        """distill → recall returns cached entry even without quro-memory."""
        bridge = QuroMemoryBridge()
        bridge._get_api = lambda: None  # type: ignore[method-assign]

        sig = self._sig()
        result = bridge.distill(sig, 0, body="round 0 body", tags=["t1"])
        # distill returns memory_id (not None — cache is always written).
        assert result is not None

        # recall reads from cache.
        recalled = bridge.recall(sig)
        assert len(recalled) == 1
        assert recalled[0]["body"] == "round 0 body"

    def test_recall_with_type_filter(self) -> None:
        bridge = QuroMemoryBridge()
        bridge._get_api = lambda: None  # type: ignore[method-assign]

        sig = self._sig()
        bridge.distill(sig, 0, body="fact entry", memory_type="fact")
        bridge.distill(sig, 1, body="episode entry", memory_type="episode")

        facts = bridge.recall(sig, memory_types=["fact"])
        assert len(facts) >= 1
        assert facts[0]["body"] == "fact entry"

    def test_recall_patterns_from_cache(self) -> None:
        bridge = QuroMemoryBridge()
        bridge._get_api = lambda: None  # type: ignore[method-assign]

        # Use a unique sig to avoid cross-test cache contamination.
        sig = ProblemSignature.compute("recall_patterns_test", ["x"])
        bridge.distill(
            sig, 0, body="stalled", tags=["pattern:goal_unmet", "round:0"],
        )
        bridge.distill(
            sig, 1, body="ok", tags=["pattern:solved", "round:1"],
        )

        results = bridge.recall_patterns(["pattern:goal_unmet"])
        assert len(results) >= 1
        # The last result should be our cached entry.
        cached = [r for r in results if r.get("body") == "stalled"]
        assert len(cached) == 1

    def test_protocol_compliance(self) -> None:
        bridge: IMemoryBridge = QuroMemoryBridge()
        bridge._get_api = lambda: None  # type: ignore[method-assign]
        assert bridge.recall(self._sig()) == []
