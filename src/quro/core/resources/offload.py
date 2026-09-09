"""Payload offload — keep ``DomainState`` references, not payloads (architecture §7).

``DomainState`` stays a plain JSON-safe dict (kernel contract, unchanged) but
holds **references**: large payload fields (artifact body, clue text, report
chunk, message history) move to ``IResourceStore``, and the state keeps the
descriptor + summary.  Reads resolve through ``IResourceStore.load`` on access
— nothing is preloaded, the OS page cache owns hot pages.
"""

from __future__ import annotations

from typing import Any

from quro.core.resources.refs import ResourceRef
from quro.core.resources.store import IResourceStore

# The in-state marker key pointing at the offloaded payload's descriptor.
OFFLOAD_REF_KEY = "ref"


def offload_payload(
    store: IResourceStore,
    ref: ResourceRef,
    descriptor: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Offload *payload* to *store*; return the reference entry for the state.

    *descriptor* holds the small in-state fields (id, summary, …); *payload*
    holds the offloaded body fields.  The returned entry keeps the descriptor
    plus a ``ref`` marker — the payload is no longer resident in ``DomainState``.
    """
    store.save(ref, payload)
    entry = dict(descriptor)
    entry[OFFLOAD_REF_KEY] = str(ref)
    return entry


def resolve_offloaded(
    store: IResourceStore,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a reference entry on access: load its payload from *store*.

    Returns the descriptor merged with the stored payload (the offloaded body
    is materialised only for the caller).  Falls back to *entry* when there is
    no ``ref`` marker or the payload is missing (weak, dangling-tolerant).
    """
    ref_str = entry.get(OFFLOAD_REF_KEY)
    if not ref_str:
        return entry
    from quro.core.resources.refs import parse_descriptor

    try:
        stored = store.load(parse_descriptor(ref_str))
    except Exception:
        stored = None
    if stored is None:
        return entry
    merged = {k: v for k, v in entry.items() if k != OFFLOAD_REF_KEY}
    for key, value in stored.items():
        merged[key] = value
    return merged
