---
name: report-template
description: Final research report skeleton for codebase exploration.
---

# report-template

Assemble the final research report from the per-phase chunks at the SUCCESS
CRITERIA path. The report must carry `file:line` evidence for every claim.

## Skeleton

```
# <research question>

## Summary
<2-4 sentences answering the question>

## Findings
### <finding title>
- evidence: <file>:<line>
- claim: <one-line claim>

## Evidence index
| file | line | claim |
|---|---|---|

## Open questions
<what remains unresolved and why>
```

## Rules

- Every Finding must cite at least one `file:line` evidence record.
- Merge per-phase report chunks (do not rewrite them) into this skeleton.
- Write the report file, then record it with
  `add_artifact(kind="file", body=<path>)` and `assemble_report(path=<path>)`.
