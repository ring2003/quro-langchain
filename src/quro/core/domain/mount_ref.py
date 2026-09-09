"""MountRef — the author-declared address of a foreign, frozen pipeline.

reuse-mount (``mount-semantics.md`` §5): a domain author declares, on a
``StepType``, that this step's body is a foreign frozen pipeline.
``MountRef`` names that pipeline so the runtime can resolve its definite
``[StepConfig]`` sequence at mount time (freeze-then-flatten, §3.3).

``MountRef`` is pure data (JSON-safe): it carries no live pipeline object,
no registry reference, no execution handle — exactly like ``Feature`` and
``skill_pool``, it is a name-only declaration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MountRef:
    """A ``(domain, pipeline)`` address of a foreign frozen pipeline."""

    domain: str
    pipeline: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"domain": self.domain, "pipeline": self.pipeline}

    @classmethod
    def from_dict(cls, data: dict[str, str] | None) -> "MountRef | None":
        """Rebuild from a JSON-loaded dict (round-trip of :meth:`to_dict`).

        ``None`` / empty data yields ``None`` (an unmounted step).
        """
        if not data:
            return None
        return cls(
            domain=str(data.get("domain", "")),
            pipeline=str(data.get("pipeline", "")),
        )
