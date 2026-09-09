"""StepCatalogAdapter — IStepCatalogView implementation wrapping StepTypeCatalog.

Phase 2 deliverable (architecture §3.4).
"""

from __future__ import annotations

from typing import Any

from quro.core.protocols import IStepCatalogView, OperatorSummary


class StepCatalogAdapter(IStepCatalogView):
    """Concrete ``IStepCatalogView`` backed by a ``StepTypeCatalog``.

    Delegates ``describe_operators()`` to the catalog's own method.
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def describe_operators(self) -> list[OperatorSummary]:
        return self._catalog.describe_operators()
