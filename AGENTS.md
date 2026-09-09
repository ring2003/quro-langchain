# AGENTS.md

## Hard Constraints

2. **MCP Utilization**:
   - Always query docs MCP for explored knowledge to narrow down problem (the doc probably not update to date).
   - Always query MCP tools first for codebase context before accessing files directly.

3. **Language**:
   - All documentation and source code MUST be written in English.

4. **Confirm User Semantics First**:
   - Whenever the user describes a problem, immediately confirm the intended
     semantics with the user. NEVER guess or assume what the user means.

5. **Experimental Project — Follow User Instructions**:
   - This is an experimental project: every task focuses on the user's core
     request. When the user asks to clean up / delete legacy code and unit
     tests, do not be overly cautious — execute the user's instructions.

6. **Challenge the User's Decisions — Independently**:
   - The user's decisions are often provisional ("off the top of my head").
     Treat every user decision as a *hypothesis to be tested*, not a settled
     conclusion. Independently, objectively, and rigorously examine the user's
     reasoning; you have a responsibility to correct it when it is wrong,
     mis-attributed, or conflates distinct concepts.
   - Disagree with evidence, not deference; silent agreement is a failure.
     When a decision is examined and confirmed, then follow it.
   - Scope: this constraint governs *judgment* (is the user's design/reasoning
     correct?); constraint 5 governs *execution* (carry out a confirmed
     instruction without over-caution). The two do not conflict.

## Design Decisions — g/f Separation (locked; do not re-argue)

These are canonical. Implement against them; delete stale code that contradicts
them — do not add backward-compatible shims. Full semantics live in
`blueprint.md` and `architecture/primitive-step.md` §4; this section is the
operative summary only.

1. **Two orthogonal operators.**
   - `f : ExecUnit → Artifact` — the execution operator. Strictly closed: `f`
     emits an Artifact (or material that folds into one), **never** `[StepSpec]`.
   - `g : Problem → [StepSpec]` — the planning/decomposition operator. Pluggable
     and optional (`g = identity` when the domain author writes `[StepSpec]`
     literally, i.e. a first-class pipeline; `g = HTN solver` or `g = LLM
     planner` when decomposition must be dynamic).

2. **Closure law** (what makes a pipeline a first-class citizen):
   `f(step) === f(pipeline)`. `ExecUnit` spans a single step and a whole
   pipeline/subtree; nesting depth is transparent to the caller.
   Consequence: **no step may emit `[StepSpec]`**. A `plan` step that emits
   `[StepSpec]` makes `f`'s output type equal `f`'s input type and breaks
   closure — that, not "step dependencies", is demo14's "PEF missing planner"
   root cause.

3. **Both decomposition forms are kept — NOT interchangeable.**
   - **outer `g() → f()`** — the meta-planner owns top-level exploration rounds
     and narrows via round goals. Compression = round-boundary checkpoints
     (`RoundCheckpoint`); message transport = working-memory / escalation
     journal; replay cost = high (cross-round).
   - **inner `f()` composite-step dynamic objective** — a composite step (e.g.
     `explore`) narrows its own objective inside its own session, emitting
     objective text (Artifact-bound), via the `Primitive` one-shot inlining
     mechanism. Compression = the step's own session; message transport = step
     session context; replay cost = low (within-step).
   They differ in compression layer, message-transport layer, and replay cost
   by orders of magnitude. The architecture keeps BOTH; neither may subsume the
   other.

4. **Stale mechanisms to delete** (they contradict 1–3):
   - generator-mode `plan` step emitting `[StepSpec]`; the `PlanToSteps` type
     converter; the legacy static-schedule fallback in `pef_to_pipeline`.
   - HIL `role="human"` (`steps/hil_step.py:63`) → explicit `hil` flag.
     (`StepSpec.role` / `StepType.role` / `StepType.capability` were already
     removed in Phase 0 — do not re-add them.)
   - `request_decomposition` op — a live but stub op (only `sub_steps`
     `setdefault`, no real decomposition) wired through `phase_ops.py:27`,
     `core/tools/coordinator.py:42`, and both `workflow_domain.py` and
     `codebase_research/domain.py:381`. Delete it in one pass, not piecemeal.
   - `EngineeringWorkflowDomain` fallback wiring in the runtime
     (`_is_planner_step`, `_planner_domain` vs `_executor_domain` split in
     `runtime/adapter.py`).

5. **Correct mechanisms to keep / generalize:**
   - `Primitive` (`src/quro/runtime/primitive.py`) — one-shot kernel-free
     inlining emitting next-instruction text. This is the correct shape for
     inner-`f` dynamic objective; generalize it beyond recovery/backtrack to the
     `explore` composite-step narrowing loop.
   - `RecoveryHook` + recovery journal — step-level retry (kept).

## Blueprint Documentation Structure

The `docs/unsat-policy-loop/` directory tracks the UNSAT-driven policy loop
blueprint and its phased implementation.

```
docs/architecture
├── STATUS.md              ← Minimal navigation index (canonical + architecture
│                            + demos + research). Historical records live in git.
├── blueprint.md           ← Canonical architecture blueprint (two-tier solver,
│                            UNSAT semantics, escalation ladder). Source of truth
│                            for design decisions — do not fork or restate.
├── working-memory.md      ← Working-memory / compressed-state design.
├── demo4/                 ← E2E verification demo for Phase 1+2.
│   ├── problem.txt        ← Task spec with goal_facts in SUCCESS CRITERIA.
│   └── run_demo.py        ← Launcher: PolicyLoopRunner + goal_facts +
│                            evaluate + compact (inline roles, no YAML).
└── research/              ← Feasibility research, mainstream-implementation
                             comparison, risk mitigation.
```

Maintenance rules:

- **Demo numbering is sequential** (demo10 = next). Each demo is a
  self-contained directory under `docs/unsat-policy-loop/demo<N>/`; demos
  that validate a specific phase link back to that phase's architecture doc.
- **Protocol-driven / adapter pattern** is mandatory for integration
  modules (memory, algorithm, planner): consumers depend on ABC protocols
  (`IWorkingMemory`, `IMemoryBridge`), never on concrete backends.
- **Algorithm layer** (`src/quro/algorithm/`) must stay pure — zero
  dependencies, independently testable math. Upper layers adapt, never
  couple.

