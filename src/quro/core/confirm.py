"""User confirmation for model switches.

When the Controller switches between agent roles (Planner, Worker, Evaluator)
that use different backends, it pauses to request user confirmation before
making the first API call with the new backend.
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Optional


def confirm_model_switch(
    *,
    task_id: str,
    from_model: Optional[str],
    to_model: str,
    auto_confirm: bool = False,
    timeout: float = 3600.0,
) -> bool:
    """Display switch info and wait for user confirmation.

    Returns True if auto_confirm or user agrees, False to abort.

    In non-TTY environments (CI, CI/CD pipelines), always returns True
    after a short timeout to avoid blocking.

    Args:
        task_id: Identifier for this task, format "role/phase/step_id".
        from_model: The previously used model name, or None for first use.
        to_model: The new model name.
        auto_confirm: If True, skip confirmation and return True immediately.
        timeout: Seconds to wait for user input in TTY mode (default: 3600, i.e. 1 hour).

    Returns:
        True if the switch is allowed, False to abort.
    """
    # Auto mode: skip confirmation entirely
    if auto_confirm:
        return True

    # Non-TTY: print info and auto-continue after timeout
    if not sys.stdin.isatty():
        print(f"\n── 模型切换 ─────────────────────────────────────")
        print(f"  Task ID : {task_id}")
        if from_model:
            print(f"  Current : {from_model} → {to_model}")
        else:
            print(f"  Model   : {to_model}")
        print(f"  Confirm : Auto-continue (no TTY, {timeout}s timeout)")
        print(f"──")
        # Brief delay for user to see the message
        time.sleep(0.5)
        return True

    # TTY: interactive mode
    print(f"\n── 模型切换 ─────────────────────────────────────")
    print(f"  Task ID : {task_id}")
    if from_model:
        print(f"  Current : {from_model} → {to_model}")
    else:
        print(f"  Model   : {to_model}")

    # Set up timeout thread
    result = [True]  # default: auto-continue on timeout

    def _timeout_handler():
        time.sleep(timeout)
        result[0] = True  # timeout → auto-continue

    timeout_thread = threading.Thread(target=_timeout_handler, daemon=True)
    timeout_thread.start()

    # Wait for user input in a separate thread to not block the main thread
    user_input = [None]

    def _input_handler():
        try:
            user_input[0] = input(f"\n  Proceed? [y/N] ").strip().lower()
        except EOFError:
            user_input[0] = "n"

    input_thread = threading.Thread(target=_input_handler, daemon=True)
    input_thread.start()

    # Wait for either timeout or user input
    input_thread.join(timeout=timeout + 1)

    if user_input[0] is not None:
        # User responded
        timeout_thread.join(timeout=0.1)
        if user_input[0] in ("y", "yes"):
            print(f"  Proceed: YES")
            return True
        else:
            print(f"  Proceed: NO — aborting")
            return False
    else:
        # Timeout — auto-continue
        print(f"\n  Confirm : Auto-continue after {timeout}s timeout")
        print(f"──")
        return True
