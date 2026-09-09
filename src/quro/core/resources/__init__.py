"""Unified resource layer — logical descriptors, graph, ACL, projection grants.

Phase-1 slice (implementation plan ``docs/unsat-policy-loop/architecture/
unified-resource-layer-implementation-plan.md``):

    - ``refs``     — ``ResourceRef`` + ``parse_descriptor`` (logical descriptors)
    - ``resolver`` — ``DomainStateResolver`` (descriptor → DomainState projection)
    - ``graph``    — ``DomainStateGraph`` (the shared edge table)
    - ``acl``      — ``DefaultAclEngine`` (Layer 2 data surface, default deny)
    - ``grants``   — ``ProjectionGrant`` / ``ProjectionGrantRegistry``
    - ``store``    — ``IResourceStore`` / ``FileResourceStore`` (phase 2 stage 1)
    - ``job``      — ``JobKey`` / ``JobIndex`` (phase 2 stage 2)
    - ``runtime_session`` — ``RuntimeSessionLedger`` (phase 2 stage 3)
    - ``resume``   — event replay + resume (phase 2 stage 5)
    - ``offload``  — payload offload (low memory, phase 2 stage 6)
"""

from __future__ import annotations

from quro.core.resources.acl import (
    Decision,
    DefaultAclEngine,
    IAclEngine,
    Principal,
    Verdict,
    acl_for_state,
    allow,
    deny,
    principal_from_state,
)
from quro.core.resources.grants import (
    IProjectionGrant,
    ProjectionGrant,
    ProjectionGrantRegistry,
)
from quro.core.resources.graph import (
    DEPENDS_ON,
    OWNS,
    DomainStateGraph,
    IResourceGraph,
)
from quro.core.resources.job import (
    InvalidJobDescriptor,
    JobIndex,
    JobKey,
    goal_uuid_from_facts,
    parse_job_descriptor,
    validate_problem_name,
)
from quro.core.resources.mount import (
    IMountAdapter,
    NaiveCopyMountAdapter,
    NoopMountAdapter,
)
from quro.core.resources.offload import (
    OFFLOAD_REF_KEY,
    offload_payload,
    resolve_offloaded,
)
from quro.core.resources.refs import (
    RESOURCE_KINDS,
    InvalidDescriptor,
    ResourceRef,
    parse_descriptor,
)
from quro.core.resources.resolver import (
    DomainStateResolver,
    IResourceResolver,
    LivePolicy,
    ResourceNotFound,
)
from quro.core.resources.resume import (
    apply_event,
    replay_events,
    resume_domain_state,
)
from quro.core.resources.runtime_session import RuntimeSessionLedger
from quro.core.resources.store import FileResourceStore, IResourceStore

__all__ = [
    "DEPENDS_ON",
    "OWNS",
    "RESOURCE_KINDS",
    "Decision",
    "DefaultAclEngine",
    "DomainStateGraph",
    "DomainStateResolver",
    "FileResourceStore",
    "IAclEngine",
    "IMountAdapter",
    "IProjectionGrant",
    "IResourceGraph",
    "IResourceResolver",
    "IResourceStore",
    "InvalidDescriptor",
    "InvalidJobDescriptor",
    "JobIndex",
    "JobKey",
    "LivePolicy",
    "NaiveCopyMountAdapter",
    "NoopMountAdapter",
    "OFFLOAD_REF_KEY",
    "Principal",
    "ProjectionGrant",
    "ProjectionGrantRegistry",
    "ResourceNotFound",
    "ResourceRef",
    "RuntimeSessionLedger",
    "Verdict",
    "acl_for_state",
    "allow",
    "apply_event",
    "deny",
    "goal_uuid_from_facts",
    "offload_payload",
    "parse_descriptor",
    "parse_job_descriptor",
    "principal_from_state",
    "replay_events",
    "resolve_offloaded",
    "resume_domain_state",
    "validate_problem_name",
]
