import os
import warnings
import re

# load environment variables before other imports
from deepeval.config.settings import autoload_dotenv, get_settings

autoload_dotenv()

from ._version import __version__
from deepeval.evaluate import evaluate, assert_test
from deepeval.evaluate.compare import compare
from deepeval.test_run import on_test_run_end, log_hyperparameters
from deepeval.utils import login
from deepeval.telemetry import *


settings = get_settings()
if not settings.DEEPEVAL_GRPC_LOGGING:
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "")


__all__ = [
    "login",
    "log_hyperparameters",
    "evaluate",
    "assert_test",
    "on_test_run_end",
    "compare",
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
    return os.getenv("DEEPEVAL_UPDATE_WARNING_OPT_IN") == "1"


def is_read_only_env():
    return os.getenv("DEEPEVAL_FILE_SYSTEM") == "READ_ONLY"


if update_warning_opt_in():
    check_for_update()
