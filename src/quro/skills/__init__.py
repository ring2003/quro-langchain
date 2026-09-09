"""Skill resources — Anthropic-style ``.md`` resources (Phase B).

A ``skill`` is an independent markdown resource shared across roles, loaded
on demand, and never cached.  Its canonical format is the Anthropic skill
format: a YAML head (``name`` + ``description``) followed by the skill body.

See ``docs/unsat-policy-loop/architecture/evolution.md`` §2.2 / §2.4.
"""

from quro.skills.catalog import (
    Skill,
    SkillCatalog,
    load_skill_file,
    render_skill_block,
)

__all__ = ["Skill", "SkillCatalog", "load_skill_file", "render_skill_block"]
