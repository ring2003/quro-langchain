"""Filesystem tools for the Worker agent.

These tools provide read/write/edit/search operations
over the local filesystem, enabling the Worker to interact
with project files during task execution.
"""

from __future__ import annotations

import os
import re
import subprocess
from glob import iglob
from pathlib import Path

from langchain_core.tools import tool


@tool
def read(path: str) -> str:
    """Read the full content of a file at the given path. Path is relative to the current working directory (".")."""
    try:
        content = Path(path).read_text(encoding="utf-8")
        return content
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except IsADirectoryError:
        return f"Error: path is a directory: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
def write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed. Path is relative to the current working directory (".")."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


@tool
def ls(path: str = ".") -> str:
    """List directory contents. Defaults to current working directory (".")."""
    try:
        entries = os.listdir(path)
        entries.sort()
        lines = []
        for e in entries:
            full = os.path.join(path, e)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(f"{e}{suffix}")
        return "\n".join(lines) if lines else "(empty)"
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except NotADirectoryError:
        return f"Error: not a directory: {path}"
    except Exception as e:
        return f"Error listing {path}: {e}"


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a file. Path is relative to the current working directory (".")."""
    try:
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            return f"Error: string not found in {path}"
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully replaced one occurrence in {path}"
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error editing {path}: {e}"


@tool
def grep(pattern: str, path: str = ".") -> str:
    """Search for a regex pattern in files (recursive if path is a directory). Returns matching lines with file:line:content. Path defaults to the current working directory (".")."""
    try:
        p = Path(path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob("*"))
            files = [f for f in files if f.is_file()]
        else:
            return f"Error: path not found: {path}"

        results = []
        compiled = re.compile(pattern)
        for f in files:
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if compiled.search(line):
                        results.append(f"{f}:{i}:{line}")
            except (UnicodeDecodeError, Exception):
                continue

        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error searching {path}: {e}"


@tool
def find(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern, recursively from path. Path defaults to the current working directory (".")."""
    try:
        results = sorted(iglob(f"{path}/**/{pattern}", recursive=True))
        if not results:
            return "(no matches)"
        return "\n".join(results)
    except Exception as e:
        return f"Error finding files: {e}"


@tool
def shell(command: str, timeout: int = 30) -> str:
    """Execute a shell command in the current working directory and return its output.

    This is an arbitrary code-execution capability (write/delete/execute) —
    it is a high-risk grant, never included in any filesystem profile.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n" + result.stderr
            else:
                output = result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"


# ---------------------------------------------------------------------------
# Filesystem tool pools and profiles
# ---------------------------------------------------------------------------
#
# Profiles are dimension-honest names a ``StepType.tools`` declaration may
# reference.  ``shell`` is deliberately **excluded** from every profile: it is
# an arbitrary code-execution grant (subprocess.run(shell=True)) and must be
# named explicitly, never folded into a "safe default".

ALL_WORKER_TOOLS = [read, write, ls, edit, grep, find, shell]

# profile name → flat tool-name list (the coordinator expands these).
PROFILES: dict[str, tuple[str, ...]] = {
    "read_only": ("read", "ls", "grep", "find"),
    "read_write": ("read", "write", "ls", "edit", "grep", "find"),
}


def all_worker_tools() -> list:  # noqa: ANN401
    """Return every filesystem tool (including the standalone ``shell``)."""
    return list(ALL_WORKER_TOOLS)
