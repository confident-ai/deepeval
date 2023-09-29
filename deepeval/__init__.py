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
import re
from ._version import __version__


def compare_versions(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

    return normalize(version1) > normalize(version2)


def check_for_update():
    try:
        import requests

        response = requests.get("https://pypi.org/pypi/deepeval/json")
        latest_version = response.json()["info"]["version"]

        if compare_versions(latest_version, __version__):
            warnings.warn(
                f'You are using deepeval version {__version__}, however version {latest_version} is available. You should consider upgrading via the "pip install --upgrade deepeval" command.'
            )
    except ModuleNotFoundError:
        # they're just getting the version
        pass


check_for_update()
