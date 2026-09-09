"""Tests for the adapter ↔ checkpoint-hook reasoning-session wiring (follow-up #1).

The ``CheckpointHook`` hands the stable kernel session id + reasoning storage
dir over via ``hook_ctx.metadata``; ``StepAdapter._build_reasoning_session``
reuses them so the kernel session is referenced (never copied) and its event
log is persisted under ``reasoning/``.
"""

from __future__ import annotations

from quro.core.domain.workflow_domain import EngineeringWorkflowDomain
from quro.core.resources import FileResourceStore
from quro.runtime.adapter import StepAdapter
from quro.runtime.tools import ToolRegistry
from quro.steps.checkpoint_hook import CheckpointHook
from quro.steps.core import StepSpec
from quro.steps.hooks import HookContext


class _Backend:
    model_name = "test-model"


def _adapter() -> StepAdapter:
    return StepAdapter(ToolRegistry(), _Backend())


def _hook_context(tmp_path) -> HookContext:
    store = FileResourceStore(tmp_path / "session")
    hook = CheckpointHook(store, session_id="s1", round_idx=0)
    ctx = HookContext(problem="p", step_results={})
    hook.on_pre_step(StepSpec(id="s2"), ctx)
    return ctx


def test_adapter_reuses_stable_reasoning_session_and_persists(tmp_path):
    ctx = _hook_context(tmp_path)
    adapter = _adapter()
    domain = EngineeringWorkflowDomain()

    session = adapter._build_reasoning_session(domain, "test-model", {"_hook_context": ctx})

    # Stable kernel session id handed over by the hook.
    assert session.metadata.session_id == "s1-s2"

    # The kernel event log + snapshot are persisted under reasoning/
    # (referenced, not copied).
    reasoning_dir = tmp_path / "session" / "reasoning"
    session.call(f"{domain.name}_init", {"problem": "p"})
    session.call("set_interpretation", {"text": "x"})
    assert (reasoning_dir / "s1-s2.checkpoint.json").exists()
    assert (reasoning_dir / "s1-s2.jsonl").exists()


def test_adapter_without_hook_keeps_in_memory_default(tmp_path):
    adapter = _adapter()
    domain = EngineeringWorkflowDomain()

    session = adapter._build_reasoning_session(domain, "test-model", {})

    # No hook → kernel default random id + no storage (in-memory).
    assert session.metadata.session_id.startswith("sess_")
    assert session._storage is None


# --------------------------------------------------------------------------
# R0b — kernel session resume (recovery-roadmap.md §3)
# --------------------------------------------------------------------------


def _hook_context_for(tmp_path, step_id: str = "s2", session_id: str = "s1") -> HookContext:
    store = FileResourceStore(tmp_path / "session")
    hook = CheckpointHook(store, session_id=session_id, round_idx=0)
    ctx = HookContext(problem="p", step_results={})
    hook.on_pre_step(StepSpec(id=step_id), ctx)
    return ctx


def test_r0b_resumes_kernel_session_when_snapshot_exists(tmp_path):
    domain = EngineeringWorkflowDomain()
    # Two independent hook contexts over the same job dir → the same stable
    # reasoning_session_id (session_id + step_id are deterministic).
    ctx_a = _hook_context_for(tmp_path)
    adapter = StepAdapter(ToolRegistry(), _Backend(), resume_kernel_sessions=True)

    # First process: fresh session, init persists the checkpoint snapshot.
    first = adapter._build_reasoning_session(domain, "test-model", {"_hook_context": ctx_a})
    assert not first.initialised
    first.call(f"{domain.name}_init", {"problem": "p"})
    first.call("checkpoint", {"label": "cp1"})
    assert first.initialised

    # Second process (restart): the snapshot is loaded and state replayed.
    ctx_b = _hook_context_for(tmp_path)
    second = adapter._build_reasoning_session(domain, "test-model", {"_hook_context": ctx_b})

    assert second.initialised, "resumed session must replay its state"
    assert second.metadata.session_id == "s1-s2"
    assert second._init_locked, "a resumed post-init session must be init-locked"
    assert len(second.log.to_list()) >= len(first.log.to_list())


def test_r0b_creates_fresh_session_when_no_snapshot(tmp_path):
    domain = EngineeringWorkflowDomain()
    ctx = _hook_context_for(tmp_path)
    adapter = StepAdapter(ToolRegistry(), _Backend(), resume_kernel_sessions=True)

    session = adapter._build_reasoning_session(domain, "test-model", {"_hook_context": ctx})

    # No checkpoint snapshot yet → fresh (not resumed), stable id still applied.
    assert not session.initialised
    assert session.metadata.session_id == "s1-s2"


def test_r0b_default_does_not_resume(tmp_path):
    domain = EngineeringWorkflowDomain()
    ctx_a = _hook_context_for(tmp_path)
    # Persist a snapshot using a resume-enabled adapter first.
    StepAdapter(ToolRegistry(), _Backend(), resume_kernel_sessions=True)._build_reasoning_session(
        domain, "test-model", {"_hook_context": ctx_a}
    ).call(f"{domain.name}_init", {"problem": "p"})

    # Default (resume_kernel_sessions=False): snapshot is ignored → fresh session.
    adapter = StepAdapter(ToolRegistry(), _Backend())
    ctx_b = _hook_context_for(tmp_path)
    session = adapter._build_reasoning_session(domain, "test-model", {"_hook_context": ctx_b})

    assert not session.initialised
