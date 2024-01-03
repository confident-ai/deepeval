import warnings
import re

# Optionally add telemtry
from ._version import __version__

from .decorators.hyperparameters import set_hyperparameters
from deepeval.event import track
from deepeval.evaluate import evaluate, run_test, assert_test
from deepeval.test_run import on_test_run_end
from deepeval.telemetry import *

__all__ = [
    "set_hyperparameters",
    "track",
    "evaluate",
    "run_test",
    "assert_test",
    "on_test_run_end",
]


def compare_versions(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

    return normalize(version1) > normalize(version2)


def check_for_update():
    try:
        import requests

        try:
            response = requests.get("https://pypi.org/pypi/deepeval/json")
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
