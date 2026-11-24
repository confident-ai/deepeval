"""Remembers the arguments of the last `deepeval test run` in a directory, so
`deepeval link-ci` can run the same tests in CI."""

import json
import os
from typing import List, Optional

from deepeval.constants import HIDDEN_DIR

LATEST_TEST_RUN_ARGS_FILE_NAME = ".latest_test_run_args.json"


def _args_file_path() -> str:
    return os.path.join(HIDDEN_DIR, LATEST_TEST_RUN_ARGS_FILE_NAME)


def save_test_run_args(args: List[str]) -> None:
    try:
        os.makedirs(HIDDEN_DIR, exist_ok=True)
        with open(_args_file_path(), "w", encoding="utf-8") as file:
            json.dump({"args": args}, file)
    except OSError:
        pass


def load_test_run_args() -> Optional[List[str]]:
    try:
        with open(_args_file_path(), encoding="utf-8") as file:
            args = json.load(file).get("args")
    except (OSError, ValueError, AttributeError):
        return None
    if (
        isinstance(args, list)
        and args
        and all(isinstance(a, str) for a in args)
    ):
        return args
    return None
