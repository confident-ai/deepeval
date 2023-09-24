# from .test_utils import assert_exact_match, TestEvalCase
# from . import (
#     _version,
#     dataset,
#     evaluator,
#     metrics,
#     query_generator,
#     test_case,
#     test_utils,
#     utils,
# )
import warnings
from ._version import __version__


def check_for_update():
    try:
        import requests

        response = requests.get("https://pypi.org/pypi/deepeval/json")
        latest_version = response.json()["info"]["version"]
        if latest_version > __version__:
            warnings.warn(
                f'You are using deepeval version {__version__}, however version {latest_version} is available. You should consider upgrading via the "pip install --upgrade deepeval" command.'
            )
    except ModuleNotFoundError:
        # they're just getting the version
        pass


check_for_update()
