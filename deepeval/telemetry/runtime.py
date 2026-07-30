"""Where deepeval is running.

Computed once at import. Known blind spots, accepted: self-hosted runners,
Modal, Databricks, and Airflow land in `script` or `container`.
"""

import os
import sys
from functools import lru_cache

from deepeval.telemetry.properties import Runtime

_CI_VENDOR_VARS = (
    "JENKINS_URL",
    "CIRCLECI",
    "BUILDKITE",
    "TF_BUILD",
    "TEAMCITY_VERSION",
    "TRAVIS",
    "APPVEYOR",
    "CI",
)


def _in_notebook() -> bool:
    # Read IPython from sys.modules rather than importing it: if it has not
    # already been imported we are certainly not inside a notebook, and this
    # avoids pulling a heavy dependency in just to answer the question.
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        shell = ipython.get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


def _in_container() -> bool:
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        return os.path.exists("/.dockerenv")
    except OSError:
        return False


def _stdin_is_a_tty() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        # stdin can be closed or replaced with a non-file object.
        return False


@lru_cache(maxsize=1)
def detect_runtime() -> Runtime:
    # Notebook first: a hosted notebook can also look like a container, and the
    # notebook fact is the more useful one.
    if _in_notebook():
        return Runtime.NOTEBOOK
    if os.environ.get("GITHUB_ACTIONS"):
        return Runtime.CI_GITHUB
    if os.environ.get("GITLAB_CI"):
        return Runtime.CI_GITLAB
    if any(os.environ.get(var) for var in _CI_VENDOR_VARS):
        return Runtime.CI_OTHER
    if _in_container():
        return Runtime.CONTAINER
    if _stdin_is_a_tty():
        return Runtime.INTERACTIVE
    return Runtime.SCRIPT
