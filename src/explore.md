# Codebase Exploration Findings

## Architecture Overview

The project implements a **multi-agent workflow engine** (MVP-2) with a clear layered architecture:

```
Domain Layer          →  quro/core/domain/
Controller Layer      →  quro/core/controller.py
Phase Machine         →  quro/core/phase_machine.py
Budget Auditor        →  quro/core/budget_auditor.py
Session Tools         →  quro/core/session/tools.py
Worker Tools          →  quro/core/worker_tools.py
Artifact Store        →  quro/core/artifact_store.py
Graph Layer           →  quro/graph/
Runtime Layer         →  quro/runtime/
Prompts               →  quro/prompts/
```

---

## Core Components

### 1. Phase Machine (`phase_machine.py`)
- **Phases**: understanding → step_spec → step_execute → step_execute:subtree → evaluate → done
- **Phase transitions** triggered by tool calls:
  - `confirm_understanding` → step_spec
  - `finalize_step` → step_execute
  - `complete_step` → evaluate
  - `submit_evaluation(complete)` → done
  - `submit_evaluation(needs_work)` → step_spec (rollback)
  - `worker_request_decomposition` → step_execute:subtree
  - `fold_tree` → step_execute
- **Allowed ops per phase**: enforced via `PHASE_OPS` dict

### 2. Controller (`controller.py`)
- Orchestrates the entire workflow loop
- **Key responsibilities**:
  - Phase management via PhaseMachine
  - Idle detection (consecutive same phase + same step_id)
  - Per-step retry limits (`max_retries_per_step`)
  - Phase escalation limits (`max_phase_escalations`)
  - Idle round limits (`max_idle_rounds`)
  - Subtree decomposition (worker_request_decomposition)
  - Budget auditing
  - Metrics collection
- **Config**: `Mvp2Config` with max_rounds=12, context_budget_chars=12000, etc.

### 3. Session Tools (`session/tools.py`)
- **Planner tools**: set_interpretation, set_plan, confirm_understanding, create_step, add_step_field, set_step_depends_on, set_step_access, set_step_hint_max_chars, finalize_step
- **Worker tools**: add_artifact, get_artifact, get_step, complete_step, terminate_subtree, checkpoint, backtrack, request_decomposition, resolve_clarification
- **Session calls**: delegate to `ReasoningSession.call()` with domain operation validation

### 4. Worker Tools (`worker_tools.py`)
- Filesystem tools built with `@tool` decorator (langchain_core.tools)
- **Explorer tools** (read-only): read, ls, grep, find, bash
- **Builder tools** (read-write): read, write, ls, edit, grep, find, bash

### 5. Domain Layer (`core/domain/workflow_domain.py`)
- `EngineeringWorkflowDomain` extends `quro_thinking.kernel.Domain`
- `initial_state()` returns initial domain state with phase, problem, steps, etc.
- `apply(state, op, args)` validates phase-appropriate operations and applies state mutations
- **Step overlap detection**: prevents creating steps with overlapping objectives/expected_output

### 6. Graph Layer (`graph/`)
- Built on **LangGraph** (`StateGraph`)
- **Nodes**: planner, explorer, builder
- **Conditional edges**: planner routes to explorer or builder based on `worker_type`
- **State**: `ReasoningState` TypedDict with problem, reasoning_text, status, next_task, etc.

### 7. Runtime Layer (`runtime/`)
- `IRuntimeBackend` Protocol: `complete()`, `complete_with_meta()`, `complete_one()`
- `OpenAICompatBackend`: raw OpenAI SDK with reasoning_content support
- `FakeBackend`: canned responses for testing
- `StreamAccumulator`: accumulates streamed content + reasoning tokens

### 8. Artifact Store (`artifact_store.py`)
- `IArtifactStore` Protocol: `save()`, `get()`, `list()`
- `FileArtifactStore`: JSON files per artifact_id in session directory

### 9. Prompts (`prompts/`)
- `PromptTemplateLoader`: Jinja2-based template resolution
- Template directories: planner, explorer, builder, auditor, controller/{phase}
- Falls back to built-in defaults if QURO_PROMPT_DIR not set

---

## Key Patterns

### Tool Calling (ReAct Loop)
- Backends run ReAct loops when tools are provided
- Tool results are fed back to the model
- Tool calls formatted as JSON in assistant messages

### Step-Based Workflow
- Steps have: step_id, objective, inputs, expected_output, validation, dependencies, checklist
- Steps can depend on other steps (depends_on)
- Steps have access modes (hints) for pull-based artifact retrieval
- Steps can be decomposed into sub-steps (request_decomposition)

### Phase-Based State Machine
- Current phase determines allowed operations
- Phase transitions triggered by specific tool call patterns
- Phase machine detects transitions and updates state

### Budget Auditing
- BudgetAuditor evaluates proposed subtree decompositions
- Considers: completion percentage, budget used/remaining, current tree state
- Returns ACCEPT/REJECT decision

### Worker Roles
- **Explorer**: read-only, uses read/ls/grep/find/bash tools
- **Builder**: read-write, uses all explorer tools + write/edit tools
- Role determined by planner's TASK: line prefix ([explorer] or [builder])

---

## File Inventory (by module)

| Module | Key Files |
|--------|-----------|
| core | worker.py, worker_tools.py, controller.py, protocols.py, types.py, phase_machine.py, budget_auditor.py, artifact_store.py, fold.py, reasoning.py |
| core/domain | workflow_domain.py |
| core/session | tools.py |
| core/integration | projector.py |
| core/mcp_tools | mcp_tools.py |
| graph | build.py, edges.py, nodes.py, state.py |
| runtime | base.py, openai_compat.py, display.py |
| prompts | loader.py |
| app | run.py, run_mvp2.py |
| config | settings.py |

---

## Integration Points

### quro_thinking.kernel
- `ReasoningSession`: main session class with domain, tool calling, checkpointing, folding
- `Domain`: base domain with initial_state() and apply()
- `DomainState`, `ToolResponse`, `VerifyResult`

### quro.core.session.tools
- `make_all_tools()`: creates StructuredTool instances bound to session calls
- `add_artifact()`: records work with artifact_id, summary, kind, body, evidences

### quro.core.worker
- `ExplorerSession`: read-only session with EXPLORER_TOOLS
- `BuilderSession`: read-write session with BUILDER_TOOLS

### MCP Tools
- `MCPToolSession`: wraps MCP protocol for tool integration

### Projector
- `project_state()`: projects domain state for display

---

## Scope for Implementation

The codebase is fully functional as an MVP-2 multi-agent workflow engine. Key areas for extension:

1. **New roles**: Add custom role system prompts (currently hardcoded role_map in controller)
2. **Additional backends**: Implement more IRuntimeBackend implementations
3. **Enhanced evaluation**: Extend evaluate phase with more sophisticated evaluation
4. **Subtree decomposition**: Worker can decompose steps into sub-steps
5. **Checkpointing/backtracking**: Already implemented via quro_thinking.kernel
6. **Budget auditing**: BudgetAuditor evaluates subtree decompositions
7. **Display**: Runtime display with reasoning streaming
