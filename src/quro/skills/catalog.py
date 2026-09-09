"""``SkillCatalog`` — load and index Anthropic-style skill ``.md`` resources.

A skill file looks like::

    ---
    name: docx
    description: Generate and edit Word documents.
    ---
    # body ...

The YAML head is parsed for ``name`` and ``description``; everything after
the closing ``---`` is the skill *body*.  ``SkillCatalog`` exposes
description-only hints (``describe``) and the full body (``body_of``) so the
runtime can render ``description``-only SKILL BLOCK hints and load the body
on demand — never pre-injecting it (evolution.md §2.4).

The loader is pure data + filesystem IO; it has no kernel or runtime deps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class Skill:
    """A single skill resource.

    Attributes:
        name: Skill identifier (from the YAML head ``name``).
        path: Source file path (for ``- [name](path)`` hints).
        description: One-line description (from the YAML head).
        body: Full skill body (everything after the YAML head).
    """

    name: str
    path: str = ""
    description: str = ""
    body: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return ``{name, path, description}`` (quro-thinking FR-3 shape)."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
        }


def _parse_frontmatter(
    text: str, fallback_name: str
) -> tuple[str, str, str]:
    """Split a skill file into ``(name, description, body)``.

    A leading ``---`` opens the YAML head; the next ``---`` closes it.  A
    missing or unparseable head falls back to ``fallback_name`` with an empty
    description, treating the whole file as the body.
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            head, body = parts[1], parts[2].strip()
            meta: dict[str, Any] = {}
            if yaml is not None and head.strip():
                try:
                    parsed = yaml.safe_load(head)
                    if isinstance(parsed, dict):
                        meta = parsed
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("unparseable skill frontmatter in %s: %s", fallback_name, exc)
            name = str(meta.get("name") or fallback_name).strip()
            description = str(meta.get("description") or "").strip()
            return name, description, body
    return fallback_name, "", text.strip()


def load_skill_file(path: str | Path) -> Skill:
    """Load one skill file from *path*."""
    p = Path(path)
    name, description, body = _parse_frontmatter(
        p.read_text(encoding="utf-8"), fallback_name=p.stem
    )
    return Skill(name=name, path=str(p), description=description, body=body)


class SkillCatalog:
    """Index of skills by name, loaded from a directory of ``.md`` files."""

    def __init__(self, skills: Iterable[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self._skills[skill.name] = skill

    @classmethod
    def from_dir(cls, directory: str | Path) -> "SkillCatalog":
        """Build a catalog from every ``*.md`` file in *directory*."""
        path = Path(directory)
        skills: list[Skill] = []
        if path.is_dir():
            for file in sorted(path.glob("*.md")):
                try:
                    skills.append(load_skill_file(file))
                except OSError as exc:
                    logger.warning("could not load skill %s: %s", file, exc)
        return cls(skills)

    def get(self, name: str) -> Skill | None:
        """Return the skill named *name*, or None."""
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        """Return True when *name* is a known skill."""
        return name in self._skills

    def names(self) -> list[str]:
        """Return all skill names (insertion order)."""
        return list(self._skills)

    def describe(self, name: str) -> str | None:
        """Return a one-line ``name: description`` hint, or None."""
        skill = self._skills.get(name)
        if skill is None:
            return None
        if skill.description:
            return f"{skill.name}: {skill.description}"
        return skill.name

    def body_of(self, name: str) -> str | None:
        """Return the full skill body for *name*, or None."""
        skill = self._skills.get(name)
        return skill.body if skill is not None else None

    def metadata(self) -> list[dict[str, Any]]:
        """Return ``[{name, path, description}, ...]`` for every skill.

        Mirrors quro-thinking's ``Domain.get_skill_metadata()`` contract
        (FR-3) so a domain can forward its catalog to the runtime.
        """
        return [skill.to_metadata() for skill in self._skills.values()]


def render_skill_block(
    declared: list[str] | None,
    pool: list[str] | None,
    catalog: SkillCatalog | None,
) -> str:
    """Render the SKILL block for prompt assembly (evolution.md §2.4).

    - **Declared skills** (the planner's ``skills⊆pool``) render as a
      ``REQUIRED`` tips block: ``- must load skill(<name>)``.
    - **Undeclared skills** (the rest of the StepType pool) render as weak
      hints: ``- [<name>](<path>)`` one-liners.

    Skill bodies are **not** pre-injected; the runtime loads them on demand
    via ``load_skill``.  Returns an empty string when there is nothing to
    render.
    """
    declared_set = {str(s).strip() for s in (declared or []) if str(s).strip()}
    pool_list = [str(s).strip() for s in (pool or []) if str(s).strip()]
    catalog = catalog or SkillCatalog()

    lines: list[str] = []
    required = [s for s in pool_list if s in declared_set]
    if required:
        lines.append("## REQUIRED SKILLS")
        lines.extend(f"- must load skill({name})" for name in required)

    hinted = [s for s in pool_list if s not in declared_set]
    if hinted:
        hint_lines: list[str] = []
        for name in hinted:
            skill = catalog.get(name)
            if skill is not None and skill.path:
                hint_lines.append(f"- [{name}]({skill.path})")
            else:
                hint_lines.append(f"- {name}")
        if hint_lines:
            lines.append("## SKILL HINTS")
            lines.extend(hint_lines)

    return "\n".join(lines)

