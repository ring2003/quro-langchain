"""Skill tools — on-demand skill body injection (Phase B).

The declarative semantic tools moved to :mod:`quro.core.tools` (toolset +
atomic-operation annotation).  This module keeps only ``make_skill_tools``,
which is not a domain op — it reads the skill catalog on demand.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool


def make_skill_tools(skill_catalog: Any = None) -> list[StructuredTool]:
    """Build the runtime skill tools (Phase B).

    ``load_skill`` pulls a skill *body* on demand and injects it as its own
    delimited block (``## SKILL: <name>``) — never pre-injected into the
    main prompt (evolution.md §2.4).  Returns an empty list when no skill
    catalog is available.
    """
    if skill_catalog is None:
        return []

    def load_skill(name: str) -> str:
        """Load the full body of a skill by name. Returns it as a delimited
        ``## SKILL: <name>`` block for on-demand guidance."""
        skill = skill_catalog.get(name)
        if skill is None:
            known = ", ".join(skill_catalog.names()) or "(none)"
            return f"Error: unknown skill '{name}'. Available: {known}"
        return f"## SKILL: {skill.name}\n\n{skill.body}"

    return [StructuredTool.from_function(func=load_skill)]
