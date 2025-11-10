from __future__ import annotations

import logging
import os
import re
import warnings

# IMPORTANT: load environment variables before other imports
from deepeval.config.settings import autoload_dotenv, get_settings

logging.getLogger("deepeval").addHandler(logging.NullHandler())
autoload_dotenv()


def _expose_public_api() -> None:
    # All other imports must happen after env is loaded
    # Do not do this at module level or ruff will complain with E402
    global __version__, evaluate, assert_test, compare
    global on_test_run_end, log_hyperparameters, login, telemetry

    from ._version import __version__ as _version
    from deepeval.evaluate import (
        evaluate as _evaluate,
        assert_test as _assert_test,
    )
    from deepeval.evaluate.compare import compare as _compare
    from deepeval.test_run import (
        on_test_run_end as _on_end,
        log_hyperparameters as _log_hparams,
    )
    from deepeval.utils import login as _login
    import deepeval.telemetry as _telemetry

    __version__ = _version
    evaluate = _evaluate
    assert_test = _assert_test
    compare = _compare
    on_test_run_end = _on_end
    log_hyperparameters = _log_hparams
    login = _login
    telemetry = _telemetry


_expose_public_api()


settings = get_settings()

if not settings.DEEPEVAL_GRPC_LOGGING:
    if os.getenv("GRPC_VERBOSITY") is None:
        os.environ["GRPC_VERBOSITY"] = settings.GRPC_VERBOSITY or "ERROR"
    if os.getenv("GRPC_TRACE") is None:
        os.environ["GRPC_TRACE"] = settings.GRPC_TRACE or ""


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


if update_warning_opt_in():
    check_for_update()
