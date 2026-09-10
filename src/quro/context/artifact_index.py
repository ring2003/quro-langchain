"""ArtifactIndexView — the ONE artifact projection (architecture §5 artifact view).

Previously three independent block renderers each re-derived the *own vs
dependency* split, the dedup, and the ACL grant check from the raw session
state by hand:

- ``own_history`` — the step agent's own prior-round artifacts
  (``step_execute`` phases).
- ``hints`` — ``access='hints'`` dependency artifacts, as a preview table
  (``step_execute`` phases).
- ``steering_artifacts`` — the steering act's full own+dep bodies.

Hand-rolled copies of the same projection are the "shadow implementation"
defect (project defect class #5): they can drift apart in their split, their
dedup, or in which artifacts they grant.  :func:`build_artifact_index`
collapses them into a single pure projection; the three renderers become thin
formatters over its entries instead of independent implementations.

The projection answers one question — *what does the current step-instance see
and with what access?* — and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quro.core.resources.refs import ResourceRef


@dataclass(frozen=True)
class ArtifactRef:
    """A compact, summary-only reference to one artifact (id-addressable)."""

    id: str
    step_id: str
    kind: str
    summary: str


@dataclass(frozen=True)
class ArtifactEntry(ArtifactRef):
    """One artifact in the view: summary plus its full body + evidence."""

    body: str = ""
    evidences: list[Any] = field(default_factory=list)

    def to_ref(self) -> ArtifactRef:
        return ArtifactRef(self.id, self.step_id, self.kind, self.summary)


@dataclass
class ArtifactIndex:
    """The canonical projection: deduped, ACL-filtered artifact entries.

    ``artifacts`` is the union of ``own`` (this step's own artifacts) then
    ``dependency`` (dependency artifacts the principal is granted read access
    to), de-duplicated by ``id`` in that order.  ``artifacts_by_kind`` buckets
    the same entries by ``kind`` for kind-addressable lookup.
    """

    artifacts: list[ArtifactEntry] = field(default_factory=list)
    artifacts_by_kind: dict[str, list[ArtifactEntry]] = field(
        default_factory=dict
    )

    @property
    def own(self) -> list[ArtifactEntry]:
        """Artifacts owned by the viewing step-instance."""
        sid = self._owner_id()
        return [e for e in self.artifacts if e.step_id == sid]

    @property
    def dependency(self) -> list[ArtifactEntry]:
        """Dependency artifacts (owned by another step, read-granted)."""
        sid = self._owner_id()
        return [e for e in self.artifacts if e.step_id != sid]

    @property
    def artifact_ids(self) -> list[str]:
        return [e.id for e in self.artifacts]

    def _owner_id(self) -> str:  # pragma: no cover - placeholder, overridden
        return ""


@dataclass
class ArtifactIndexView:
    """The projection plus the bookkeeping a prompt needs to speak about it.

    ``artifact_count`` is the total number of artifacts the principal can see
    (own + granted dependency) — the single, honest count the model must
    address by id rather than guess at an opaque number (anti-patterns H/I).

    ``dep_artifact_ids`` lets a formatter / rule surface *which* ids are
    dependencies (so a step can be told "these bodies are upstream, cite them,
    do not rewrite them").
    """

    index: ArtifactIndex
    principal_id: str
    artifact_count: int
    dep_artifact_ids: list[str] = field(default_factory=list)

    @property
    def artifacts(self) -> list[ArtifactEntry]:
        return self.index.artifacts

    @property
    def own(self) -> list[ArtifactEntry]:
        return self.index.own

    @property
    def dependency(self) -> list[ArtifactEntry]:
        return self.index.dependency

    def _owner_id(self) -> str:  # pragma: no cover
        return self.principal_id


# Patch the owner-id accessor onto the instance the view wraps so ``own`` /
# ``dependency`` split against the correct step id.
def _attach_owner(view: ArtifactIndexView) -> None:
    index = view.index

    def _owner() -> str:
        return view.principal_id

    index._owner_id = _owner  # type: ignore[attr-defined]


@runtime_checkable
class IArtifactIndex(Protocol):
    """Read surface of the projection that formatters depend on."""

    @property
    def artifacts(self) -> list[ArtifactEntry]: ...

    @property
    def artifacts_by_kind(self) -> dict[str, list[ArtifactEntry]]: ...


def _is_artifact_entry(a: Any) -> bool:
    return isinstance(a, dict) and bool(a.get("artifact_id"))


def build_artifact_index(
    state: dict[str, Any] | None,
    principal: Any,  # Principal (quro.core.resources.acl)
    acl: Any,  # IAclEngine (DefaultAclEngine or None)
) -> ArtifactIndexView:
    """Pure projection: what *principal* sees in *state*'s artifacts.

    Args:
        state: The raw ``DomainState`` dict (``state["artifacts"]``).
        principal: The requesting step-instance (``Principal``).  Its
            ``step_id`` selects own artifacts.
        acl: The ACL engine for the dependency grant check.  When ``None`` the
            dependency set is empty (no grants resolved) — the projection is
            still well-formed (own artifacts only).

    Returns:
        An :class:`ArtifactIndexView`.  Own artifacts come first (in state
        order, de-duped); dependency artifacts follow, each included only when
        ``acl.check(principal, artifact_ref, "read").allowed``.
    """
    all_a = [
        a
        for a in (state.get("artifacts") if isinstance(state, dict) else [])
        if _is_artifact_entry(a)
    ]
    step_id = principal.step_id if principal is not None else ""

    # Own artifacts — this step's artifacts, deduped by id, stable order.
    seen: set[str] = set()
    own: list[ArtifactEntry] = []
    for a in all_a:
        aid = a["artifact_id"]
        if a.get("step_id") == step_id and aid not in seen:
            seen.add(aid)
            own.append(_entry(a))

    # Dependency artifacts — another step owns them; include only when the
    # ACL grants read.  The DefaultAclEngine enforces owner ∈ depends_on and
    # access == "hints" at once.
    dependency: list[ArtifactEntry] = []
    if principal is not None and acl is not None:
        for a in all_a:
            aid = a["artifact_id"]
            if aid in seen:
                continue
            try:
                verdict = acl.check(
                    principal, ResourceRef("artifact", aid), "read"
                )
            except Exception:
                # An unresolvable ref (e.g. no owning step) is a deny, never a
                # crash — deny wins.
                continue
            if verdict.allowed:
                seen.add(aid)
                dependency.append(_entry(a))

    entries = own + dependency
    by_kind: dict[str, list[ArtifactEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)

    index = ArtifactIndex(artifacts=entries, artifacts_by_kind=by_kind)
    view = ArtifactIndexView(
        index=index,
        principal_id=step_id,
        artifact_count=len(entries),
        dep_artifact_ids=[e.id for e in dependency],
    )
    _attach_owner(view)
    return view


def build_global_artifact_index(
    artifact_dicts: list[dict[str, Any]],
) -> ArtifactIndexView:
    """Build the MetaPlanner / orchestrator-wide artifact projection.

    ``build_artifact_index`` is scoped to a single step-instance principal:
    it splits artifacts into *own* and ACL-granted *dependency*.  The
    MetaPlanner sits *above* the steps — it legitimately sees every artifact
    across all rounds and steps, so ownership and ACL grants do not apply.

    This is the same canonical :class:`ArtifactIndexView` record shape the
    step agents consume (same ``ArtifactEntry`` projection via ``_entry``),
    with all artifacts read-granted.  It keeps "one semantic view, multiple
    prompt consumers" — the consumer differs (an orchestrator vs a step), not
    the artifact record semantics.

    Args:
        artifact_dicts: The raw artifact dicts (each with ``artifact_id``).

    Returns:
        An :class:`ArtifactIndexView` containing every dict as an entry
        (de-duplicated by id, stable order).
    """
    seen: set[str] = set()
    entries: list[ArtifactEntry] = []
    for a in artifact_dicts:
        aid = a.get("artifact_id") or a.get("id")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        entries.append(_entry(a))

    by_kind: dict[str, list[ArtifactEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)

    index = ArtifactIndex(artifacts=entries, artifacts_by_kind=by_kind)
    view = ArtifactIndexView(
        index=index,
        principal_id="__meta_planner__",
        artifact_count=len(entries),
        dep_artifact_ids=[e.id for e in entries],
    )
    _attach_owner(view)
    return view


def _entry(a: dict[str, Any]) -> ArtifactEntry:
    evidences = a.get("evidences")
    if isinstance(evidences, list):
        ev = [str(x) for x in evidences]
    else:
        ev = [str(evidences)] if evidences else []
    return ArtifactEntry(
        id=a.get("artifact_id", "") or a.get("id", ""),
        step_id=a.get("step_id", ""),
        kind=a.get("kind", ""),
        summary=str(a.get("summary", "") or "").strip(),
        body=str(a.get("body", "") or "").strip(),
        evidences=ev,
    )
