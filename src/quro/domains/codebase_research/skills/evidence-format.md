---
name: evidence-format
description: The compact evidence record shape (file:line + one-line claim).
---

# evidence-format

Evidence is deposited compactly and structurally, never as full file text.
Each evidence record is one finding at one location.

## Record shape

- `file` — the source file path (relative to the codebase root when possible).
- `line` — the line number or line range (e.g. `42` or `120-135`).
- `claim` — one sentence describing what the code at that location does or
  implies, tied to the research question.

## Rules

- One claim per record. Split compound findings into multiple records.
- `line` must point at real code, not a comment or blank line.
- Prefer exact ranges over whole-file references.

## Example

```
file: src/moe/expert_cache.cpp
line: 230-255
claim: expert cache lookup path routes through ExpertCache::find(), which is
       the single insertion point for cache population.
```
