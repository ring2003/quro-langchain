---
name: docx
description: Create, read, and edit Microsoft Word (.docx) documents.
---

# docx

Work with Word documents. Use `python-docx` to create and edit `.docx`
files, and read text content back for verification.

## Workflow

1. Inspect any existing document before editing it.
2. Make the requested change (add/replace text, tables, or styles).
3. Save the file and read it back to confirm the change landed.
4. Record the produced file path via `add_artifact(kind="file", body=...)`.

## Constraints

- Preserve existing formatting unless the task asks to change it.
- Never overwrite the source document without a backup path.
