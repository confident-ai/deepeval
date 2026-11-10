"""Offline replay and regression checks for CI, with no provider calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .artifacts import ArtifactLimits, read_artifact, write_artifact
from .comparison import compare_reports
from .engine import evaluate
from .reports import Verdict


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse may include a supplied filename/value in its message. Avoid
        # echoing arbitrary evidence on malformed CLI invocations.
        self.print_usage(sys.stderr)
        self.exit(3, "Invalid execution-contract command arguments.\n")


def _positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser():
    parser = _Parser(
        prog="python -m deepeval.metrics.community.execution_contract",
        description="Replay deterministic execution contracts without an LLM judge.",
        epilog="Exit codes: 0 pass/no regression, 1 failure/regression, 2 unknown, 3 invalid input or I/O.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser(
        "replay", help="Evaluate a saved snapshot under a saved contract"
    )
    replay.add_argument("--contract", required=True, type=Path)
    replay.add_argument("--evidence", required=True, type=Path)
    replay.add_argument(
        "--output", type=Path, help="Atomically write a report artifact"
    )
    replay.add_argument(
        "--max-bytes",
        type=_positive_integer,
        default=ArtifactLimits().max_bytes,
    )
    compare = commands.add_parser(
        "compare", help="Compare reports from the same contract"
    )
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument(
        "--max-bytes",
        type=_positive_integer,
        default=ArtifactLimits().max_bytes,
    )
    return parser


def _protect_inputs(destination, sources):
    if destination is None:
        return
    resolved = destination.resolve()
    for source in sources:
        if resolved == source.resolve():
            raise ValueError("report output must differ from input files")
        if (
            destination.exists()
            and source.exists()
            and os.path.samefile(destination, source)
        ):
            raise ValueError("report output aliases an input file")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    limits = ArtifactLimits(max_bytes=args.max_bytes)
    try:
        if args.command == "replay":
            _protect_inputs(args.output, (args.contract, args.evidence))
            contract = read_artifact(
                args.contract, expected_kind="contract", limits=limits
            )
            evidence = read_artifact(
                args.evidence, expected_kind="snapshot", limits=limits
            )
            report = evaluate(evidence, contract)
            if args.output is not None:
                write_artifact(args.output, report, limits=limits)
            output = report.to_dict()
            exit_code = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.UNKNOWN: 2}[
                report.status
            ]
        else:
            baseline = read_artifact(
                args.baseline, expected_kind="report", limits=limits
            )
            candidate = read_artifact(
                args.candidate, expected_kind="report", limits=limits
            )
            comparison = compare_reports(baseline, candidate)
            output, exit_code = comparison.to_dict(), comparison.exit_code
        print(
            json.dumps(
                output, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
        return exit_code
    except (ValueError, TypeError, OSError, RecursionError):
        # Library APIs preserve typed errors for debugging; a CI log should not
        # leak raw file content or paths via exception repr/JSON parse excerpts.
        print(
            "Execution-contract command failed: invalid input, resource limit, or file I/O.\n"
            "Use the library API locally for detailed validation errors.",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
