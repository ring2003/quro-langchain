---
name: read-strategy
description: How to judge code hot-spots and read files efficiently under a bounded context.
---

# read-strategy

Reading a codebase under a bounded context means never pasting whole files.
Judge hot-spots first, then read the smallest slice that answers the current
question.

## Workflow

1. Start from the directory structure / module index, not from file contents.
2. Identify the 2–3 files most likely to answer the round goal (entry points,
   definitions, call sites).
3. Open a file and read only the relevant range (definition, key function
   body), never the whole file.
4. Extract one finding at a time and persist it immediately: `commit_clue`
   for incremental persistence, then assemble the step's evidence with
   `add_artifact` (evidence lives in the artifact, `file:line` + one-line
   claim).

## Constraints

- Prefer structural navigation (`codegraph`) over textual `grep` for
  definition / reference / call-graph questions.
- If a file is too large to read in full, record the explored range and the
  unresolved question as evidence instead of looping.
- Do not commit a large accumulated block of findings at the end — commit
  small findings frequently.
