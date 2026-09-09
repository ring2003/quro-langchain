---
name: codegraph
description: Structured codebase exploration via the codegraph CLI (files/query/explore/node/callers/callees/impact).
---

# codegraph

Structured codebase exploration via the `codegraph` CLI (v1.5.0). Run commands
through the `bash` tool — no resident server, no thin wrapper. Load this body
on demand and compose the exact command you need, then turn each finding into
a compact evidence record (`file:line` + one-line claim) via `add_artifact` /
`commit_clue`.

`-p/--path` defaults to the current directory; pass it explicitly when the
target codebase is elsewhere.

## Index lifecycle

```bash
codegraph index [path]    # full (re)index — run once before exploring
codegraph sync [path]     # incremental sync since last index
codegraph status [path]   # index status/statistics (-j for JSON)
```

If exploration returns empty or stale results, run `codegraph sync` (or
`codegraph index` when the index is missing).

## Structure

```bash
codegraph files -p <path>                # file tree (default)
codegraph files -p <path> --format flat  # flat list
codegraph files -p <path> --filter src/  # under a directory
codegraph files -p <path> --pattern '*.cpp'
```

Use `files` first to orient; never `ls`/`find` blindly when the index already
knows the structure.

## Symbol search

```bash
codegraph query 'expert cache'            # semantic/name search
codegraph query 'ExpertCache' -k class    # filter by node kind
codegraph query 'find' -l 20 -j           # more results, JSON
```

`query` returns symbols with their locations; use it to go from a concept to
the defining files.

## One-shot exploration

```bash
codegraph explore <query...>              # symbols + source + call paths in one
codegraph node <name>                     # one symbol's source + caller/callee trail
codegraph node -f <file> --offset N --limit M   # read a file range with line numbers
codegraph node -f <file> --symbols-only   # just the symbol map + dependents
```

`explore` / `node` are the fastest way to pull a bounded slice: the relevant
source plus its call paths, instead of pasting whole files.

## Call graph

```bash
codegraph callers <symbol> -l 20   # who calls <symbol>
codegraph callees <symbol> -l 20   # what <symbol> calls
codegraph impact <symbol> -d 2     # what a change to <symbol> affects
codegraph affected src/a.cpp       # test files affected by changed sources
```

Use `callers`/`callees` to trace a data-flow question; use `impact`/`affected`
to bound the blast radius.

## Rules

- One exploration question per command; keep the output slice small.
- Extract the `file:line` + claim immediately after each command — do not
  accumulate raw command output in context.
- If a result is too large, narrow with `--limit`, `--offset/--limit`, or
  `--filter` rather than pasting everything.
