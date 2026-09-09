"""Core StepSpec and StepResult dataclasses.

A ``StepSpec`` represents a declarative step definition — what the step type
does, which tools it needs, and how it chains to other steps.

A ``StepResult`` captures the outcome of executing a ``StepSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from quro.steps.hooks import StepHook

# ---------------------------------------------------------------------------
# StepSpec
# ---------------------------------------------------------------------------

@dataclass
class StepSpec:
    """A declarative step definition.

    Unlike the current system where steps are created by the LLM at
    runtime via ``create_step``, this dataclass allows pre-defining
    steps with explicit tool requirements, dependencies, and validation
    rules.

    Args:
        id: Unique step identifier (e.g. ``"understand_problem"``).
        name: Human-readable name for the step.
        objective: What this step should accomplish.
        tools: Names of tools available to this step. Empty list means
               only domain tools.
        depends_on: List of step IDs this step depends on.
        expected_output: Description of the expected output.
        validation: Validation rule for step completion.
        timeout: Maximum execution time in seconds.
        inputs: Required input artifacts from dependencies.
        checklist: Items to verify before completion.
        sub_steps: Sub-steps that can be executed within this step.
        hil: Human-in-the-loop flag (Phase 0: replaces ``role == "human"``).
    """

    id: str
    name: str = ""
    objective: str = ""
    tools: list[Any] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    expected_output: str = ""
    validation: str = ""
    timeout: int = 0  # 0 means no timeout
    inputs: list[str] = field(default_factory=list)
    checklist: list[dict[str, Any]] = field(default_factory=list)
    sub_steps: list["StepSpec"] = field(default_factory=list)
    pre_hooks: list["StepHook"] = field(default_factory=list)
    post_hooks: list["StepHook"] = field(default_factory=list)
    message_strategy: str = "default"
    """Cross-round message management strategy.

    ``"default"`` — inherit from the StepType or PipelineConfig.
    ``"rebuild"`` — rebuild [system, user] each round (current behavior).
    ``"accumulate"`` — keep message history across rounds; only break on
    terminal tool (multi-step conversation).
    """
    hil: bool = False
    """Human-in-the-loop flag — pause and wait for user input."""
    step_type: str = ""
    """The StepType this step instantiates (Phase B).  Declares the skill
    pool the planner may draw from."""
    skills: list[str] = field(default_factory=list)
    """Skills declared by the planner (a subset of the StepType's pool).
    Rendered as REQUIRED tips; the rest of the pool renders as weak hints."""
    grant_name: str | None = None
    """Name of a dynamic grant callable (``GrantRegistry``).  Mirrored from
    the StepType at assembly; resolved at evaluate time and never serialized
    (only this name is)."""
    exclude: tuple[Any, ...] = field(default_factory=tuple)
    """Capability types denied from the composed grant (``static ∪ dynamic −
    exclude``).  Mirrored from the StepType at assembly."""
    features: tuple[str, ...] = field(default_factory=tuple)
    """Framework-owned step-level capabilities enabled for this step,
    mirrored from the StepType at assembly (see
    ``unified-resource-layer-phase4-implementation-plan.md`` §2.5).  Name-only
    and JSON-safe; the effect binding is the runtime consumer's contract."""
    mounted: Any = None
    """Optional ``MountRef`` declaring this step's body is a foreign, frozen
    pipeline (reuse-mount, ``mount-semantics.md`` §5).  Mirrored from the
    StepType at assembly.  ``None`` = an ordinary step (no mount).  ``Any``
    (not ``MountRef | None``) avoids a module-level import cycle: the concrete
    type lives in ``core/domain/mount_ref.py`` and is mirrored here by value."""
    next_instruction: str = ""
    """Transient next-step instruction inlined by a compaction act on resume
    (``recovery_mode="compaction"``).  Rides the prompt, not the state — it is
    deliberately NOT serialized into ``StepConfig`` (stateless, decision B in
    ``primitive-step.md`` §5)."""

    def get_tools(self) -> list[Any]:
        """Return the tool grants (capabilities / names) declared for this step."""
        return list(self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for YAML/config round-trip.

        ``tools`` capability types are serialized by name (``"Readonly"``), and
        ``exclude`` likewise; ``grant_name`` is already a name string.
        """
        from quro.core.tools.grant import capabilities_to_names

        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "tools": capabilities_to_names(self.tools),
            "depends_on": self.depends_on,
            "expected_output": self.expected_output,
            "validation": self.validation,
            "timeout": self.timeout,
            "inputs": self.inputs,
            "checklist": self.checklist,
            "hil": self.hil,
            "step_type": self.step_type,
            "skills": self.skills,
            "grant_name": self.grant_name,
            "exclude": capabilities_to_names(self.exclude),
            "features": list(self.features),
            "mounted": self.mounted.to_dict() if self.mounted is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepSpec:
        """Create a StepSpec from a dict (e.g. loaded from YAML).

        ``tools`` / ``exclude`` names are rebuilt into capability types via the
        ``CAPABILITY_BY_NAME`` map.
        """
        from quro.core.tools.grant import resolve_capabilities

        step = cls(
            id=data["id"],
            name=data.get("name", ""),
            objective=data.get("objective", ""),
            tools=list(resolve_capabilities(data.get("tools", []))),
            depends_on=data.get("depends_on", []),
            expected_output=data.get("expected_output", ""),
            validation=data.get("validation", ""),
            timeout=data.get("timeout", 0),
            inputs=data.get("inputs", []),
            checklist=data.get("checklist", []),
            message_strategy=data.get("message_strategy", "default"),
            hil=bool(data.get("hil", False)),
            step_type=data.get("step_type", ""),
            skills=data.get("skills", []),
            grant_name=data.get("grant_name"),
            exclude=resolve_capabilities(data.get("exclude", [])),
            features=tuple(data.get("features", [])),
            mounted=_mount_ref_from_dict(data.get("mounted")),
        )
        return step


def _mount_ref_from_dict(data: Any) -> Any:
    """Rebuild a ``MountRef`` from a JSON dict (``None`` when absent/empty)."""
    if not data:
        return None
    from quro.core.domain.mount_ref import MountRef

    return MountRef.from_dict(data)

# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing a ``StepSpec``.

    Args:
        step_id: The step that was executed.
        ok: Whether the step completed successfully.
        state: The domain state after step execution.
        artifacts: Artifacts produced by the step.
        error: Error message if the step failed.
        metrics: Metrics collected during execution.
        replan_request: When set, the step is asking the planner to
            reconsider the remaining plan. Contains ``reason``,
            ``suggested_new_steps``, ``suggested_removals``, and
            ``priority`` fields.
    """

    step_id: str
    ok: bool
    state: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    replan_request: dict[str, Any] | None = None

    @classmethod
    def success(cls, step_id: str, state: dict[str, Any], artifacts: list[dict[str, Any]] | None = None) -> StepResult:
        """Create a successful step result."""
        return cls(
            step_id=step_id,
            ok=True,
            state=state,
            artifacts=artifacts or [],
        )

    @classmethod
    def failure(cls, step_id: str, error: str, state: dict[str, Any] | None = None) -> StepResult:
        """Create a failed step result."""
        return cls(
            step_id=step_id,
            ok=False,
            state=state,
            error=error,
        )

    @classmethod
    def with_replan(cls, step_id: str, reason: str, *, ok: bool = True, state: dict[str, Any] | None = None, suggested_new_steps: list[dict[str, Any]] | None = None, suggested_removals: list[str] | None = None, priority: str = "normal") -> StepResult:
        """Create a result that requests replanning.

        Args:
            step_id: The step that was executed.
            reason: Why replanning is requested.
            ok: Whether the step itself succeeded.
            state: The domain state after step execution.
            suggested_new_steps: Steps the worker suggests adding.
            suggested_removals: Step IDs the worker suggests removing.
            priority: ``"low"``, ``"normal"``, or ``"high"``.
        """
        return cls(
            step_id=step_id,
            ok=ok,
            state=state,
            replan_request={
                "reason": reason,
                "suggested_new_steps": suggested_new_steps or [],
                "suggested_removals": suggested_removals or [],
                "priority": priority,
            },
        )


# ---------------------------------------------------------------------------
# StepExecutor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StepExecutor(Protocol):
    """Executes a single step given accumulated pipeline context.

    The context dict carries the overall problem and the results of steps
    executed so far (including their artifacts).
    """

    def execute(self, step: StepSpec, context: dict[str, Any]) -> StepResult:
        """Run *step* and return its result.

        Args:
            step: The step to execute.
            context: Keys include ``problem`` (str) and ``results``
                (dict[str, StepResult] of previously executed steps).
        """
        ...
