import os
import warnings
import re

# Optionally add telemetry
from ._version import __version__

from deepeval.feedback import collect_feedback, a_collect_feedback
from deepeval.evaluate import evaluate, assert_test
from deepeval.test_run import on_test_run_end, log_hyperparameters
from deepeval.utils import login_with_confident_api_key
from deepeval.telemetry import *
from deepeval.confident import confident_evaluate


if os.getenv("DEEPEVAL_GRPC_LOGGING") != "YES":
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""

__all__ = [
    "login_with_confident_api_key",
    "log_hyperparameters",
    "track",
    "a_collect_feedback",
    "collect_feedback",
    "evaluate",
    "assert_test",
    "on_test_run_end",
    "confident_evaluate",
]


def compare_versions(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

    return normalize(version1) > normalize(version2)


def check_for_update():
    try:
        import requests

        try:
            response = requests.get(
                "https://pypi.org/pypi/deepeval/json", timeout=5
            )
            latest_version = response.json()["info"]["version"]

            if compare_versions(latest_version, __version__):
                warnings.warn(
                    f'You are using deepeval version {__version__}, however version {latest_version} is available. You should consider upgrading via the "pip install --upgrade deepeval" command.'
                )
        except (
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.SSLError,
            requests.exceptions.Timeout,
        ):
            # when pypi servers go down
            pass
    except ModuleNotFoundError:
        # they're just getting the versions
        pass


def update_warning_opt_in():
    return os.getenv("DEEPEVAL_UPDATE_WARNING_OPT_IN") == "YES"


def is_read_only_env():
    return os.getenv("DEEPEVAL_FILE_SYSTEM") == "READ_ONLY"


if update_warning_opt_in():
    check_for_update()
