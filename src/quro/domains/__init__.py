"""User-authored domain extensions.

Concrete ``Domain`` plugins live here, separate from the kernel infrastructure
in ``quro.core.domain`` (``phase_ops`` / ``step_type``).  Each domain package
is self-contained: its ``domain.py`` plus its skill resources.
"""

from quro.domains.codebase_research.domain import CodebaseResearchDomain

__all__ = ["CodebaseResearchDomain"]
