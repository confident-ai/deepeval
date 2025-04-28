import os
import re
import warnings

from deepeval.confident import confident_evaluate
from deepeval.evaluate import assert_test, evaluate
from deepeval.event import track
from deepeval.monitor import a_monitor, a_send_feedback, monitor, send_feedback
from deepeval.telemetry import *
from deepeval.test_run import log_hyperparameters, on_test_run_end
from deepeval.utils import login_with_confident_api_key

# Optionally add telemetry
from ._version import __version__

__all__ = [
    "login_with_confident_api_key",
    "log_hyperparameters",
    "track",
    "monitor",
    "a_monitor",
    "a_send_feedback",
    "send_feedback",
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
        # they're just getting the versione
        pass


def update_warning_opt_in():
    return os.getenv("DEEPEVAL_UPDATE_WARNING_OPT_IN") == "YES"


if update_warning_opt_in():
    check_for_update()
