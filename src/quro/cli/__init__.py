"""``quro`` CLI package — the phase-3 read-only ``show`` command (no kernel).

The CLI does exactly three things: resolve ``state_dir``, construct the
read-only handles (``JobIndex`` / ``RuntimeSessionLedger`` /
``FileResourceStore``) from that single root, delegate to ``CLIView``, and
print.  No storage, no kernel import, no write path.

Command shape::

    quro show job://{domain}/{problem_name}/{goal_uuid}
    quro show job://… --session s1
    quro show job://… --session s1 --round 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quro.cli.view import CLIView, JobNotFoundError, SessionNotFoundError
from quro.config.settings import Settings
from quro.core.resources import (
    FileResourceStore,
    InvalidJobDescriptor,
    JobIndex,
    RuntimeSessionLedger,
    parse_job_descriptor,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the ``quro`` argument parser (``show`` is the only command)."""
    parser = argparse.ArgumentParser(
        prog="quro",
        description="Quro-Thinking resource CLI (read-only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    show = subparsers.add_parser(
        "show",
        help="Print a problem-status summary addressed by descriptor.",
        description="Print a problem-status summary addressed by a job descriptor.",
    )
    show.add_argument(
        "descriptor",
        help="job://{domain}/{problem_name}/{goal_uuid}",
    )
    show.add_argument(
        "--session",
        metavar="ID",
        help="descend to the session level (session id).",
    )
    show.add_argument(
        "--round",
        metavar="N",
        type=int,
        help="descend to the round level (round index); requires --session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command != "show":
        return 2
    return _run_show(args)


def _run_show(args: argparse.Namespace) -> int:
    if args.round is not None and args.session is None:
        print("error: --round requires --session", file=sys.stderr)
        return 2

    try:
        job = parse_job_descriptor(args.descriptor)
    except InvalidJobDescriptor as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # G1 — single state root: resolve state_dir once and construct both the
    # job/session ledgers and the resource store from it.
    root = Path(Settings().state_dir)
    index = JobIndex(root)
    ledger = RuntimeSessionLedger(root)
    view = CLIView()

    try:
        if args.session is None:
            text = view.render_job(job, index)
        else:
            store = FileResourceStore(ledger.session_dir(job, args.session))
            if args.round is None:
                text = view.render_session(job, args.session, ledger, store)
            else:
                text = view.render_round(store, args.round)
    except JobNotFoundError as exc:
        print(f"no such job: {exc}", file=sys.stderr)
        return 1
    except SessionNotFoundError as exc:
        print(f"no such session: {exc}", file=sys.stderr)
        return 1

    print(text)
    return 0
