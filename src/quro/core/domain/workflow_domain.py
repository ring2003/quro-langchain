from __future__ import annotations

from uuid import uuid4
from typing import Any

from quro_thinking.kernel import Domain, DomainState, ToolResponse, VerifyResult

from quro.context.blocks import phase_blocks
from quro.core.domain.phase_ops import ALLOWED_IN_ANY_PHASE, PHASE_OPS  # noqa: F401  (re-export)
from quro.core.domain.step_type import StepType, StepTypeCatalog
from quro.skills import SkillCatalog

# Keys that are safe to transfer between step sessions via carry_forward.
TRANSFERABLE_KEYS = frozenset({
    "plan", "interpretation", "steps", "artifacts",
    "executing_step_id", "designing_step_id", "evaluations",
    "clarification_requests",
})


def default_step_types() -> StepTypeCatalog:
    """Return the built-in engineering StepType catalog (Phase B).

    A single executor step type:

    - ``implement`` — the executor step: produce the requested output
      (``step_execute``, ended by ``complete_step``).  Ships with the bundled
      ``docx`` skill so the ``create_step(skills⊆pool)`` contract is
      exercisable out of the box.
    """
    return StepTypeCatalog([
        StepType(
            "implement", skill_pool=["docx"],
            identity_block=(
                "You are the **implement** step. Your responsibility: read, "
                "write, edit, and test to complete the assigned task. Focus "
                "ONLY on your assigned step."
            ),
            executor_hints={"phase": "step_execute", "terminal_tool": "complete_step"},
        ),
    ])


def default_skills() -> SkillCatalog:
    """Load the bundled demo skills from ``quro.skills/resources/``.

    Falls back to an empty catalog when the package resources are not
    available (e.g. source tree without the ``.md`` files installed).
    """
    try:
        from importlib import resources

        return SkillCatalog.from_dir(resources.files("quro.skills") / "resources")
    except Exception:  # pragma: no cover - defensive
        return SkillCatalog()


def _default_step_id(data: dict) -> str:
    phase = data.get("phase", "")
    if phase in ("step_execute", "step_execute:subtree"):
        return data.get("executing_step_id") or data.get("designing_step_id") or ""
    return data.get("designing_step_id") or data.get("executing_step_id") or ""


def _check_step_overlap(new_step: dict, other_steps: list[dict]) -> ToolResponse | None:
    new_objective = (new_step.get("objective", "") + " " +
                     " ".join(new_step.get("expected_output", []))).lower()
    if not new_objective.strip():
        return None
    for s in other_steps:
        if s.get("status") not in ("pending", "completed", "in_progress"):
            continue
        existing_objective = (s.get("objective", "") + " " +
                              " ".join(s.get("expected_output", []))).lower()
        new_words = set(new_objective.split())
        existing_words = set(existing_objective.split())
        if not new_words or not existing_words:
            continue
        overlap = new_words & existing_words
        if len(overlap) > max(len(new_words), len(existing_words)) * 0.6:
            return ToolResponse(
                ok=False,
                error="STEP_OVERLAP: new step objective significantly overlaps with a prior step",
                explanation=(
                    f"Step '{s['step_id']}' ({s.get('objective', '')}) "
                    f"already covers similar ground. Revise objective or expected_output "
                    f"to avoid overlap, or set depends_on to reuse prior work."
                ),
            )
    return None


@phase_blocks({
    "understanding": ["problem", "interpretation", "plan", "completed_steps", "objective"],
    "step_spec": ["problem", "plan", "completed_steps", "objective"],
    "step_execute": ["plan", "interpretation", "current_step", "completed_steps", "hints", "own_history", "objective"],
    "step_execute:subtree": ["plan", "interpretation", "current_step", "completed_steps", "hints", "own_history", "objective"],
    "evaluate": ["problem", "plan", "completed_steps", "artifacts", "objective"],
})
class EngineeringWorkflowDomain(Domain):
    name: str = "engineering_workflow"

    # InlineStep recovery declaration (primitive-step.md §6): the domain
    # declares its recovery need; the upper layer reads it and passes it to the
    # coordinator — the coordinator does not re-derive it from the domain.
    recovery_mode: str = "none"  # "none" | "compaction" | "mount"
    recovery_compaction: str | None = None  # when mode == "compaction"
    recovery_compaction_for_step_type: dict[str, str] = {}

    # Steering declaration (primitive-step.md §6/§7): which step types open the
    # invasive objective-override loop, keyed by the steering built-in name.
    # Invasive — the author MUST also declare a steering identity so the agent
    # knows its objective is a loop seed.  Default empty = no steering.
    steering_for_step_type: dict[str, str] = {}
    # Time-disjoint boundary (§7): how many times compaction may retry the
    # *same* objective before steering may replace it.  Domain-declared, never
    # an agent runtime guess.
    steering_retry_budget: int = 3

    def __init__(
        self,
        *,
        step_types: StepTypeCatalog | None = None,
        skills: SkillCatalog | None = None,
        recovery_mode: str = "none",
        recovery_compaction: str | None = None,
        recovery_compaction_for_step_type: dict[str, str] | None = None,
        steering_for_step_type: dict[str, str] | None = None,
        steering_retry_budget: int = 3,
    ) -> None:
        super().__init__()
        self._step_types = step_types or default_step_types()
        self._skills = skills or default_skills()
        self.recovery_mode = recovery_mode
        self.recovery_compaction = recovery_compaction
        self.recovery_compaction_for_step_type = dict(
            recovery_compaction_for_step_type or {}
        )
        self.steering_for_step_type = dict(steering_for_step_type or {})
        self.steering_retry_budget = steering_retry_budget

    @property
    def step_types(self) -> StepTypeCatalog:
        """The domain's StepType catalog (planner-context injection)."""
        return self._step_types

    @property
    def skills(self) -> SkillCatalog:
        """The domain's skill catalog (metadata + on-demand bodies)."""
        return self._skills

    def get_skill_metadata(self) -> list[dict[str, Any]]:
        """Return ``[{name, path, description}, ...]`` (quro-thinking FR-3)."""
        return self._skills.metadata()

    def phase_for_step_type(self, step_type: str) -> str:
        """Return the initial phase for *step_type* (understanding when unknown).

        Derived from the StepType's ``executor_hints["phase"]`` — the same
        seam a research domain declares (blueprint §3.2).
        """
        hints = self._step_types.executor_hints_for(step_type)
        return hints.get("phase") or "understanding"

    def initial_state(self, init_args: dict[str, Any]) -> DomainState:
        phase = str(init_args.get("phase") or "understanding")
        if phase not in PHASE_OPS:
            phase = "understanding"
        return {
            "phase": phase,
            "problem": init_args.get("problem", ""),
            "interpretation": "",
            "plan": "",
            "confirmed": False,
            "steps": [],
            "designing_step_id": None,
            "executing_step_id": None,
            "artifacts": [],
            "evaluations": [],
            "clarification_requests": [],
        }

    def apply(self, state: DomainState, op: str, args: dict[str, Any]) -> tuple[DomainState, ToolResponse]:
        data = dict(state)

        phase = data.get("phase", "understanding")
        allowed = PHASE_OPS.get(phase, set()) | ALLOWED_IN_ANY_PHASE
        if op not in allowed:
            return data, ToolResponse(
                ok=False,
                error=f"PHASE_MISMATCH: '{op}' not allowed in phase '{phase}'",
                explanation=f"Allowed ops in phase '{phase}': {', '.join(sorted(allowed))}",
            )

        if op == "set_interpretation":
            data["interpretation"] = args.get("text", "")

        elif op == "set_plan":
            data["plan"] = args.get("plan", "")

        elif op == "confirm_understanding":
            if not data.get("interpretation") or not data.get("plan"):
                return data, ToolResponse(
                    ok=False,
                    error="cannot confirm: interpretation and plan are required",
                    explanation="Call set_interpretation and set_plan first.",
                )
            data["confirmed"] = True
            data["phase"] = "step_spec"
            data["designing_step_id"] = None

        elif op == "create_step":
            step_id = args.get("step_id", f"step_{uuid4().hex[:4]}")
            steps = list(data.get("steps", []))
            for s in steps:
                if s["step_id"] == step_id:
                    return data, ToolResponse(
                        ok=False,
                        error=f"Step '{step_id}' already exists",
                        explanation=f"Step '{step_id}' is already defined (status: {s.get('status')}). Use a different step_id.",
                    )

            # Phase B: `create_step(step_type, objective, skills⊆pool)` contract.
            step_type_name = str(args.get("step_type") or "").strip()
            raw_skills = args.get("skills") or []
            if isinstance(raw_skills, str):
                skills = [x.strip() for x in raw_skills.split(",") if x.strip()]
            else:
                skills = [str(x).strip() for x in raw_skills if str(x).strip()]

            if step_type_name and not self._step_types.has(step_type_name):
                return data, ToolResponse(
                    ok=False,
                    error=f"INVALID_STEP_TYPE: '{step_type_name}'",
                    explanation=(
                        "Known step types: "
                        + (", ".join(self._step_types.names()) or "(none)")
                    ),
                )

            if skills:
                if not step_type_name:
                    return data, ToolResponse(
                        ok=False,
                        error="INVALID_SKILL: skills require a step_type",
                        explanation="Pass step_type=... when declaring skills.",
                    )
                pool = self._step_types.skill_pool_for(step_type_name)
                unknown = [sk for sk in skills if sk not in pool]
                if unknown:
                    return data, ToolResponse(
                        ok=False,
                        error=f"INVALID_SKILL: {', '.join(unknown)}",
                        explanation=(
                            f"StepType '{step_type_name}' skill pool: "
                            + (", ".join(pool) or "(empty)")
                        ),
                    )
                missing = [sk for sk in skills if not self._skills.has(sk)]
                if missing:
                    return data, ToolResponse(
                        ok=False,
                        error=f"INVALID_SKILL: skill(s) not found: {', '.join(missing)}",
                        explanation=(
                            "Known skills: "
                            + (", ".join(self._skills.names()) or "(none)")
                        ),
                    )

            # The step's execution unit is its StepType (Phase 0: the identity
            # word).  The planner never names a role — it names a step type.
            step = {
                "step_id": step_id,
                "objective": args.get("objective", ""),
                "step_type": step_type_name,
                "skills": skills,
                "inputs": [],
                "expected_output": [],
                "validation": [],
                "dependencies": [],
                "checklist": [],
                "sub_steps": [],
                "parent": args.get("parent", None),
                "status": "pending",  # DEPRECATED — kept for backward compat during transition
                "round_status": "in_progress",  # NEW: L0-owned, scoped to current governor session
                "step_status": "pending",  # NEW: L1-owned (steered) or L0-owned (ordinary)
                "depends_on": [],
                "access": None,
                "hint_max_chars": None,
            }
            steps.append(step)
            data["steps"] = steps
            data["designing_step_id"] = step_id

        elif op == "update_step":
            step_id = args.get("step_id", _default_step_id(data))
            field = args.get("field", "")
            value = args.get("value", "")
            steps = list(data.get("steps", []))
            found = False
            for s in steps:
                if s["step_id"] != step_id:
                    continue
                found = True
                if field == "checklist":
                    s.setdefault("checklist", []).append({
                        "item_id": args.get("item_id", f"ci_{uuid4().hex[:6]}"),
                        "description": value,
                        "done": False,
                    })
                elif field in ("inputs", "expected_output", "validation", "dependencies"):
                    s.setdefault(field, []).append(value)
                elif field == "depends_on":
                    raw = args.get("value", [])
                    if isinstance(raw, str):
                        s["depends_on"] = [x.strip() for x in raw.split(",") if x.strip()]
                    else:
                        s["depends_on"] = list(raw)
                elif field == "access":
                    access = args.get("value")
                    if access is not None and access not in ("hints",):
                        return data, ToolResponse(
                            ok=False,
                            error=f"INVALID_ACCESS_MODE: '{access}'",
                            explanation="Only access='hints' is supported.",
                        )
                    s["access"] = access
                elif field == "hint_max_chars":
                    hint_max = args.get("value")
                    if hint_max is not None:
                        try:
                            hint_max = int(hint_max)
                        except (ValueError, TypeError):
                            return data, ToolResponse(
                                ok=False,
                                error=f"INVALID_HINT_MAX_CHARS: '{hint_max}'",
                                explanation="hint_max_chars must be an integer.",
                            )
                    s["hint_max_chars"] = hint_max if hint_max else None
                else:
                    return data, ToolResponse(
                        ok=False,
                        error=f"INVALID_STEP_FIELD: '{field}'",
                        explanation=(
                            "Valid fields: inputs, expected_output, validation, "
                            "dependencies, checklist, depends_on, access, hint_max_chars."
                        ),
                    )
                break
            if not found:
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' not found",
                    explanation="Cannot update a non-existent step.",
                )
            data["steps"] = steps

        elif op == "finalize_step":
            steps = data.get("steps", [])
            if not steps:
                return data, ToolResponse(
                    ok=False, error="No steps created",
                    explanation="Create at least one step before finalizing.",
                )
            for s in steps:
                overlap_check = _check_step_overlap(s, [x for x in steps if x["step_id"] != s["step_id"]])
                if overlap_check is not None:
                    return data, overlap_check
            step_ids = {s["step_id"] for s in steps}
            for s in steps:
                deps = s.get("depends_on", [])
                access = s.get("access")
                for dep_id in deps:
                    if dep_id not in step_ids:
                        return data, ToolResponse(
                            ok=False,
                            error=f"DEP_NOT_FOUND: step '{s['step_id']}' depends_on unknown step '{dep_id}'",
                            explanation=f"Step '{dep_id}' does not exist. Create it first or fix depends_on.",
                        )
                if access == "hints" and not deps:
                    return data, ToolResponse(
                        ok=False,
                        error=f"INVALID_ACCESS: step '{s['step_id']}' has access='hints' but no depends_on",
                        explanation="Set depends_on=[...] when access='hints', or remove access.",
                    )
                if deps and access is None:
                    return data, ToolResponse(
                        ok=False,
                        error=f"MISSING_ACCESS: step '{s['step_id']}' has depends_on but no access mode",
                        explanation="Set access='hints' when depends_on is non-empty.",
                    )
                if access is not None and access != "hints":
                    return data, ToolResponse(
                        ok=False,
                        error=f"INVALID_ACCESS_MODE: step '{s['step_id']}' access='{access}'",
                        explanation="Only access='hints' is supported in MVP-2.",
                    )
                hint_chars = s.get("hint_max_chars")
                if hint_chars is not None and (not isinstance(hint_chars, int) or hint_chars < 80 or hint_chars > 400):
                    s["hint_max_chars"] = 160
            data["phase"] = "step_execute"
            initial_step = None
            for s in steps:
                if not s.get("depends_on", []):
                    initial_step = s["step_id"]
                    break
            if initial_step is None and steps:
                initial_step = steps[0]["step_id"]
            data["designing_step_id"] = None
            data["executing_step_id"] = initial_step

        elif op == "add_artifact":
            artifacts = list(data.get("artifacts", []))
            artifacts.append({
                "artifact_id": args.get("artifact_id", f"art_{uuid4().hex[:8]}"),
                "step_id": args.get("step_id", _default_step_id(data)),
                "kind": args.get("kind", "analysis"),
                "summary": args.get("summary", ""),
                "evidences": args.get("evidences", []),
                "body": args.get("body", ""),
            })
            data["artifacts"] = artifacts

        elif op == "complete_step":
            step_id = args.get("step_id", "")
            if not step_id:
                return data, ToolResponse(
                    ok=False,
                    error="complete_step requires explicit step_id",
                    explanation="Pass step_id=... to identify which step to complete.",
                )
            steps = list(data.get("steps", []))
            current_step = None
            for s in steps:
                if s["step_id"] == step_id:
                    current_step = s
                    break
            if current_step is None:
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' not found",
                    explanation=f"Step '{step_id}' does not exist. Use create_step first.",
                )

            # NEW: Set round_status only — this is L0's sole authority.
            # step_status is NOT touched here; L1 (or L0 for ordinary steps) owns it.
            if current_step.get("round_status") == "completed":
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' round already completed",
                    explanation="This governor round is already complete.",
                )

            step_artifacts = [a for a in data.get("artifacts", []) if a.get("step_id") == step_id]
            if not step_artifacts:
                return data, ToolResponse(
                    ok=False,
                    error=f"Step '{step_id}' has no artifacts",
                    explanation="Call add_artifact before complete_step.",
                )

            dep_step_ids = current_step.get("depends_on", [])
            access = current_step.get("access")
            # Defensive re-check only (unified-resource-layer phase 1): the
            # up-front ACL deny in the tool layer already guarantees the step
            # fetched only what it may, so this fires solely for a hand-crafted
            # state that bypasses the tool layer.  The access log is audit-only.
            if dep_step_ids and access == "hints":
                access_log = args.get("access_log", [])
                access_artifact_ids = {e[1] for e in access_log if e[0] == "artifact"}
                access_step_ids = {e[1] for e in access_log if e[0] == "step"}

                step_artifact_map = {}
                for a in data.get("artifacts", []):
                    sid = a.get("step_id", "")
                    if sid not in step_artifact_map:
                        step_artifact_map[sid] = []
                    step_artifact_map[sid].append(a)

                current_arts = step_artifact_map.get(step_id, [])
                output_text = " ".join(
                    a.get("body", "") + " " + a.get("summary", "") + " " + " ".join(a.get("evidences", []))
                    for a in current_arts
                )

                for dep_id in dep_step_ids:
                    dep_arts = step_artifact_map.get(dep_id, [])
                    if not dep_arts:
                        if dep_id not in access_step_ids:
                            return data, ToolResponse(
                                ok=False,
                                error="DEP_ACCESS_REQUIRED: step steals prior work without evidence",
                                explanation=f"Call get_step('{dep_id}') before complete_step.",
                            )
                    else:
                        fetched = any(a["artifact_id"] in access_artifact_ids for a in dep_arts)
                        step_fetched = dep_id in access_step_ids
                        cited = any(a["artifact_id"] in output_text for a in dep_arts)
                        if not (fetched or step_fetched or cited):
                            return data, ToolResponse(
                                ok=False,
                                error="DEP_ACCESS_REQUIRED: step steals prior work without evidence",
                                explanation=f"Call get_artifact/get_step for depends_on [{', '.join(dep_step_ids)}], or cite artifact ids in add_artifact.",
                            )

            # Determine whether this step is steering-controlled.
            # For steered steps, step_status is the L1-owned token — only the
            # fold may mint it, and only when the steering loop exited on a
            # Steering-approved terminal (final round declared, or budget
            # exhausted with artifacts).  For ordinary steps, L0 == L1, so
            # complete_step may mint both tokens directly.
            is_steered = bool(
                self.steering_for_step_type.get(
                    current_step.get("step_type", ""), ""
                )
            )
            for s in steps:
                if s["step_id"] == step_id:
                    s["round_status"] = "completed"
                    if not is_steered:
                        s["step_status"] = "completed"
                    break
            data["steps"] = steps
            remaining = [s for s in steps if s.get("step_status") not in ("completed",)]
            if not remaining:
                data["executing_step_id"] = None
                data["phase"] = "evaluate"
            else:
                completed_ids = {s["step_id"] for s in steps if s.get("step_status") == "completed"}
                next_step = None
                for s in remaining:
                    deps = s.get("depends_on", [])
                    if all(d in completed_ids for d in deps):
                        next_step = s
                        break
                if next_step is None:
                    next_step = remaining[0]
                data["executing_step_id"] = next_step["step_id"]

        elif op == "terminate_subtree":
            step_id = args.get("step_id", _default_step_id(data))
            steps = list(data.get("steps", []))
            for s in steps:
                if s["step_id"] == step_id:
                    s["round_status"] = "terminated"
                    break
            data["steps"] = steps

        elif op == "fold_tree":
            subtree_id = args.get("subtree_id", "")
            status = args.get("status", "completed")
            steps = list(data.get("steps", []))
            for s in steps:
                if s["step_id"] == subtree_id:
                    s["round_status"] = "folded"
                    break
            data["steps"] = steps
            phase_info = data.get("phase", "")
            if phase_info == "step_execute:subtree":
                data["phase"] = "step_execute"

        elif op == "submit_evaluation":
            evaluations = list(data.get("evaluations", []))
            evaluation = {
                "round_id": max((e.get("round_id", 0) for e in evaluations), default=0) + 1,
                "proposal": args.get("proposal", ""),
                "rationale": args.get("rationale", ""),
            }
            evaluations.append(evaluation)
            data["evaluations"] = evaluations
            proposal = evaluation["proposal"]
            if proposal == "complete":
                data["phase"] = "done"
            elif proposal in ("continue", "needs_work"):
                if proposal == "needs_work":
                    steps = list(data.get("steps", []))
                    for s in steps:
                        if s.get("status") == "pending":
                            s["status"] = "cancelled"
                    data["steps"] = steps
                    data["designing_step_id"] = None
                data["phase"] = "step_spec"

        elif op == "request_clarification":
            message = args.get("message", "")
            if not message:
                return data, ToolResponse(
                    ok=False, error="request_clarification requires 'message'"
                )
            clarifications = list(data.get("clarification_requests", []))
            clarifications.append({
                "from_phase": data.get("phase", ""),
                "from_step": _default_step_id(data),
                "message": message,
                "resolved": False,
            })
            data["clarification_requests"] = clarifications
            data["phase"] = "step_spec"

        elif op == "resolve_clarification":
            clarifications = list(data.get("clarification_requests", []))
            idx = args.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(clarifications):
                clarifications[idx]["resolved"] = True
            data["clarification_requests"] = clarifications

        elif op == "carry_forward":
            # Validated state hand-off between step sessions (§4.3).
            # Accepts sources=[{step_id, keys}] and target phase.
            sources: list[dict[str, Any]] = args.get("sources", [])
            phase = args.get("phase", "step_execute")
            if phase not in PHASE_OPS:
                return data, ToolResponse(
                    ok=False, error=f"INVALID_PHASE: '{phase}'"
                )

            for src in sources:
                src_keys = src.get("keys", [])
                if any(k not in TRANSFERABLE_KEYS for k in src_keys):
                    return data, ToolResponse(
                        ok=False,
                        error="INVALID_TRANSFER_KEY",
                        explanation=f"Allowed keys: {sorted(TRANSFERABLE_KEYS)}",
                    )

                for key in src_keys:
                    src_data = src.get(key)
                    if src_data is None:
                        continue
                    if key in ("steps",):
                        existing_ids = {s["step_id"] for s in data.get("steps", [])}
                        for s in src_data:
                            if s.get("step_id") not in existing_ids:
                                data.setdefault("steps", []).append(s)
                                existing_ids.add(s["step_id"])
                    elif key in ("artifacts",):
                        existing_ids = {a["artifact_id"] for a in data.get("artifacts", [])}
                        for a in src_data:
                            if a.get("artifact_id") not in existing_ids:
                                data.setdefault("artifacts", []).append(a)
                                existing_ids.add(a["artifact_id"])
                    elif key in ("evaluations", "clarification_requests"):
                        data.setdefault(key, []).extend(src_data)
                    elif key in ("executing_step_id", "designing_step_id"):
                        if not data.get(key):
                            data[key] = src_data
                    else:
                        # plan, interpretation — last-writer-wins
                        data[key] = src_data

            data["phase"] = phase
            data["confirmed"] = True

        # NOTE: checkpoint/backtrack are intercepted by the kernel session
        # before Domain.apply() (quro-thinking/kernel/session.py), so no
        # domain branch is needed here — they fall through to the default
        # ok=True passthrough below.

        return data, ToolResponse(ok=True, operation={op: True})

    def verify(self, state: DomainState, scope: str = "all") -> VerifyResult:
        errors = []
        if not state.get("interpretation"):
            errors.append("missing interpretation")
        if not state.get("plan"):
            errors.append("missing plan")
        terminal_statuses = {"completed", "folded", "terminated"}
        incomplete = [s for s in state.get("steps", []) if s.get("status") not in terminal_statuses]
        if incomplete:
            errors.append(f"steps not completed: {[s['step_id'] for s in incomplete]}")
        return VerifyResult(satisfied=[], violated=errors, pending=[])

    def is_complete(self, state: DomainState) -> bool:
        return (
            state.get("phase") == "done"
            and all(s.get("status") == "completed" for s in state.get("steps", []))
            and any(e.get("proposal") == "complete" for e in state.get("evaluations", []))
        )
