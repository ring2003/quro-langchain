"""Memory layer — protocols and adapters for working memory and persistence.

Protocols
---------
- ``IWorkingMemory`` — L0 in-session state accumulation
- ``IMemoryBridge``  — L1/L2 persistent memory (distillation + recall)

Implementations
---------------
- ``InMemoryWorkingMemory`` — Phase 1 dict-based accumulator
- ``NoopMemoryBridge`` — Phase 1 null adapter (safe default)
- ``QuroMemoryBridge`` — Phase 2 quro-memory adapter

Consumers depend on the **protocols**, not the concrete implementations.
"""

from quro.memory.bridge import NoopMemoryBridge, QuroMemoryBridge
from quro.memory.protocols import IMemoryBridge, IWorkingMemory
from quro.memory.working import InMemoryWorkingMemory

__all__ = [
    "IMemoryBridge",
    "IWorkingMemory",
    "InMemoryWorkingMemory",
    "NoopMemoryBridge",
    "QuroMemoryBridge",
]
