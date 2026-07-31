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


def _in_repl() -> bool:
    """Someone typing statements at a `>>>` prompt.

    Deliberately not `sys.stdin.isatty()`: a script launched from a shell also
    has a terminal on stdin, so that test labelled every laptop run
    `interactive` and left `script` meaning little more than "stdin was piped".
    `sys.ps1` exists only in a real interpreter session, and
    `sys.flags.interactive` covers `python -i`.
    """
    return hasattr(sys, "ps1") or bool(sys.flags.interactive)


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
    if _in_repl():
        return Runtime.INTERACTIVE
    return Runtime.SCRIPT
