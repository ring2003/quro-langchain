from __future__ import annotations

from uuid import uuid4
from typing import Any

from quro_thinking.kernel import ReasoningSession


def handle_fold_tree(
    session: ReasoningSession,
    subtree_id: str,
    conclusions: list[dict[str, Any]],
    status: str = "completed",
    axioms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoint_label = f"fold_{subtree_id}"

    cp_res = session.call("checkpoint", {"label": checkpoint_label})
    if not cp_res.ok:
        return {"status": "error", "reason": cp_res.error}

    if not axioms:
        axioms = _compress_conclusions(session, subtree_id, conclusions)

    for ax in axioms:
        ax.setdefault("axiom_id", f"ax_{uuid4().hex[:8]}")
        ax.setdefault("subtree_id", subtree_id)
        ax.setdefault("verified_count", 0)
        session.axiom_store.append(ax)

    session.call("fold_tree", {"subtree_id": subtree_id, "status": status})

    return {
        "status": "folded",
        "axioms": axioms,
        "checkpoint_label": checkpoint_label,
    }


def handle_terminate_subtree(
    session: ReasoningSession,
    reason: str,
    derive_axiom: str = "",
) -> dict[str, Any]:
    cp_label = f"pre_terminate_{uuid4().hex[:6]}"

    cp_res = session.call("checkpoint", {"label": cp_label})
    if not cp_res.ok:
        return {"status": "error", "reason": cp_res.error}

    if derive_axiom:
        axiom = {
            "axiom_id": f"ax_{uuid4().hex[:8]}",
            "statement": derive_axiom,
            "justification_brief": reason[:500],
            "result": "fail",
        }
        session.axiom_store.append(axiom)

    parent_cp = _find_parent_checkpoint(session)
    bt_res = session.call("backtrack", {
        "to_label": parent_cp,
        "reason": f"Subtree terminated: {reason[:200]}",
        "exhaust": True,
    })
    if not bt_res.ok:
        return {"status": "error", "reason": bt_res.error}

    return {"status": "subtree_terminated", "message": reason}


def _compress_conclusions(
    session: ReasoningSession,
    subtree_id: str,
    conclusions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    axioms: list[dict[str, Any]] = []
    domain = session.domain
    for c in conclusions:
        statement = c.get("statement", "")
        if not statement:
            continue
        justification = c.get("justification_brief", c.get("evidence", ""))
        if isinstance(justification, list):
            justification = "; ".join(str(e) for e in justification)
        confidence = c.get("confidence", 1.0)
        axiom = {
            "axiom_id": f"ax_{uuid4().hex[:8]}",
            "statement": statement,
            "justification_brief": str(justification)[:500],
            "subtree_id": subtree_id,
            "confidence": confidence,
            "verified_count": 0,
        }
        axioms.append(axiom)
    return axioms


def _find_parent_checkpoint(session: ReasoningSession) -> str:
    live = [l for l in session._checkpoint_order if l not in session.exhausted_branches]
    if not live:
        return "init"
    return live[0] if len(live) == 1 else live[-2] if len(live) >= 2 else live[0]
