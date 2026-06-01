"""`python -m deepeval.inspect [PATH]` entry point.

Mirrors `deepeval inspect [PATH]` for developers running from a checkout
without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

from deepeval.inspect import run_inspect
from deepeval.inspect.loader import (
    InspectLoadError,
    NoTracesError,
    find_latest_test_run,
)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    raw = args[0] if args else None

    try:
        if raw:
            resolved = Path(raw)
            if resolved.is_dir():
                resolved = find_latest_test_run(resolved)
        else:
            resolved = find_latest_test_run("experiments")
        run_inspect(str(resolved))
        return 0
    except FileNotFoundError as e:
        print(f"deepeval inspect: {e}", file=sys.stderr)
        return 2
    except (InspectLoadError, NoTracesError) as e:
        print(f"deepeval inspect: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
