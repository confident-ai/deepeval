import warnings
import re

# Optionally add telemtry
from ._version import __version__

from deepeval.event import track, send_feedback
from deepeval.evaluate import evaluate, assert_test
from deepeval.test_run import on_test_run_end, log_hyperparameters
from deepeval.utils import login_with_confident_api_key
from deepeval.telemetry import *
from deepeval.integrations import trace_langchain, trace_llama_index

__all__ = [
    "login_with_confident_api_key",
    "log_hyperparameters",
    "track",
    "evaluate",
    "assert_test",
    "on_test_run_end",
    "send_feedback",
    "trace_langchain",
    "trace_llama_index",
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


check_for_update()
