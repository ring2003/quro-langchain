"""StepConfig — the JSON-safe, addressable, exportable description of a step.

Phase 4 (``unified-resource-layer-phase4-implementation-plan.md`` §2) makes a
step's *configuration* a first-class resource.  Today it is scattered across
``StepSpec`` (objective / tools / depends_on / …) and ``StepType``
(identity_block / executor_hints / features); ``StepConfig`` converges both
into one serializable unit that can be snapshotted, exported, and re-mounted.

``StepConfig`` is a **snapshot, not a live object**: it is derived from
``StepSpec`` + ``StepType`` at checkpoint time and stored verbatim.  It must
not hold live hooks / registry references (only names), so it round-trips
through JSON.

The rebuild direction (``StepConfig → object``, a ``StepFactory``) landed in
phase 5 (``step_factory.py``); ``StepConfig`` is its only input (see
``unified-resource-layer-phase5-implementation-plan.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from quro.core.domain.step_type import StepType

if TYPE_CHECKING:
    from quro.core.domain.mount_ref import MountRef
    from quro.steps.core import StepSpec


@dataclass
class StepConfig:
    """The frozen executable description of a step (JSON-safe dict shape).

    Fields split into four sources:

    - from ``StepSpec``: ``id`` / ``name`` / ``step_type`` / ``objective`` /
      ``expected_output`` / ``validation`` / ``checklist`` / ``tools`` /
      ``depends_on`` / ``skills`` / ``message_strategy`` / ``hil`` /
      ``timeout`` / ``inputs``.
    - inlined from ``StepType``: ``identity_block`` / ``executor_hints`` /
      ``features`` / ``skill_pool``.
    - explicit hooks as *names*: ``pre_hooks`` / ``post_hooks``
      (``StepSpec.pre_hooks``/``post_hooks`` → ``hook.name``).
    - recursive: ``sub_steps`` (nested ``StepConfig``), with
      ``parent_path`` / ``depth`` recording the nesting.
    """

    # identity
    id: str
    name: str = ""
    step_type: str = ""

    # task
    objective: str = ""
    expected_output: str = ""
    validation: str = ""
    checklist: list[dict[str, Any]] = field(default_factory=list)

    # tool surface + structure
    tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    # dynamic grant + deny (capability-typed, name-only)
    grant_name: str | None = None
    exclude: list[str] = field(default_factory=list)

    # inlined from StepType
    identity_block: str = ""
    executor_hints: dict[str, Any] = field(default_factory=dict)
    features: tuple[str, ...] = field(default_factory=tuple)

    # runtime
    message_strategy: str = "default"
    hil: bool = False

    # fractal-reserved (empty in phase 4; recorded by sub_steps nesting in phase 5)
    parent_path: str = ""
    depth: int = 0

    # inlined from StepType (phase 5 — decision 32)
    skill_pool: list[str] = field(default_factory=list)

    # explicit hooks snapshot as *names* (phase 5 — decisions 32 + 34)
    pre_hooks: list[str] = field(default_factory=list)
    post_hooks: list[str] = field(default_factory=list)

    # from StepSpec (phase 5 — decision 32)
    timeout: int = 0
    inputs: list[str] = field(default_factory=list)
    sub_steps: list["StepConfig"] = field(default_factory=list)

    # reuse-mount: the foreign frozen pipeline this step's body expands into
    # (``mount-semantics.md`` §5).  ``None`` = no mount.
    mounted: "MountRef | None" = None

    @classmethod
    def from_step(cls, step: "StepSpec", step_type: StepType | None = None) -> "StepConfig":
        """Derive the snapshot from a ``StepSpec`` + its ``StepType`` (may be None).

        ``features`` prefers the ``StepSpec``'s mirrored tuple and falls back
        to the ``StepType``'s declaration.  ``sub_steps`` recurse with
        ``parent_path`` / ``depth`` recording the nesting; sub-steps have no
        ``StepType`` handle (the factory rebuilds them anonymously), so their
        inlined fields fall back to the ``StepSpec`` mirror.
        """
        return cls._build(step, step_type, parent_path="", depth=0)

    @classmethod
    def _build(
        cls,
        step: "StepSpec",
        step_type: StepType | None,
        *,
        parent_path: str,
        depth: int,
    ) -> "StepConfig":
        sub_steps = [
            cls._build(s, None, parent_path=step.id, depth=depth + 1)
            for s in step.sub_steps
        ]
        from quro.core.tools.grant import capabilities_to_names

        return cls(
            id=step.id,
            name=step.name,
            step_type=step.step_type,
            objective=step.objective,
            expected_output=step.expected_output,
            validation=step.validation,
            checklist=[dict(c) for c in step.checklist],
            tools=capabilities_to_names(step.tools),
            depends_on=list(step.depends_on),
            skills=list(step.skills),
            grant_name=step.grant_name,
            exclude=capabilities_to_names(step.exclude),
            identity_block=step_type.identity_block if step_type else "",
            executor_hints=dict(step_type.executor_hints) if step_type else {},
            features=(
                tuple(step.features)
                if step.features
                else (tuple(step_type.features) if step_type else ())
            ),
            message_strategy=step.message_strategy,
            hil=step.hil,
            parent_path=parent_path,
            depth=depth,
            skill_pool=list(step_type.skill_pool) if step_type else [],
            pre_hooks=[getattr(h, "name", "") for h in step.pre_hooks],
            post_hooks=[getattr(h, "name", "") for h in step.post_hooks],
            timeout=step.timeout,
            inputs=list(step.inputs),
            sub_steps=sub_steps,
            mounted=step.mounted,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (lists instead of tuples, copies)."""
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type,
            "objective": self.objective,
            "expected_output": self.expected_output,
            "validation": self.validation,
            "checklist": [dict(c) for c in self.checklist],
            "tools": list(self.tools),
            "depends_on": list(self.depends_on),
            "skills": list(self.skills),
            "grant_name": self.grant_name,
            "exclude": list(self.exclude),
            "identity_block": self.identity_block,
            "executor_hints": dict(self.executor_hints),
            "features": list(self.features),
            "message_strategy": self.message_strategy,
            "hil": self.hil,
            "parent_path": self.parent_path,
            "depth": self.depth,
            "skill_pool": list(self.skill_pool),
            "pre_hooks": list(self.pre_hooks),
            "post_hooks": list(self.post_hooks),
            "timeout": self.timeout,
            "inputs": list(self.inputs),
            "sub_steps": [s.to_dict() for s in self.sub_steps],
            "mounted": self.mounted.to_dict() if self.mounted else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepConfig":
        """Rebuild from a JSON-loaded dict (round-trip of :meth:`to_dict`)."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            step_type=data.get("step_type", ""),
            objective=data.get("objective", ""),
            expected_output=data.get("expected_output", ""),
            validation=data.get("validation", ""),
            checklist=[dict(c) for c in data.get("checklist", [])],
            tools=list(data.get("tools", [])),
            depends_on=list(data.get("depends_on", [])),
            skills=list(data.get("skills", [])),
            grant_name=data.get("grant_name"),
            exclude=list(data.get("exclude", [])),
            identity_block=data.get("identity_block", ""),
            executor_hints=dict(data.get("executor_hints", {})),
            features=tuple(data.get("features", [])),
            message_strategy=data.get("message_strategy", "default"),
            hil=bool(data.get("hil", False)),
            parent_path=data.get("parent_path", ""),
            depth=int(data.get("depth", 0)),
            skill_pool=list(data.get("skill_pool", [])),
            pre_hooks=list(data.get("pre_hooks", [])),
            post_hooks=list(data.get("post_hooks", [])),
            timeout=int(data.get("timeout", 0)),
            inputs=list(data.get("inputs", [])),
            sub_steps=[cls.from_dict(s) for s in data.get("sub_steps", [])],
            mounted=_mount_ref_from_data(data.get("mounted")),
        )


def _mount_ref_from_data(data: Any) -> "MountRef | None":
    """Rebuild a ``MountRef`` from a JSON dict (``None`` when absent/empty)."""
    from quro.core.domain.mount_ref import MountRef

    return MountRef.from_dict(data) if isinstance(data, dict) else None
