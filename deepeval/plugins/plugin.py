import pytest
import os
import json
from rich import print
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import test_run_manager, DeploymentConfigs
from deepeval.utils import get_is_running_deepeval


def pytest_sessionstart(session: pytest.Session):
    is_running_deepeval = get_is_running_deepeval()

    if is_running_deepeval:
        test_run_manager.save_to_disk = True
        try:
            deployment_configs = session.config.getoption("--deployment")
            disable_request = False

            if deployment_configs is None:
                deployment = False
            else:
                deployment = True
                deployment_configs = json.loads(deployment_configs)
                disable_request = deployment_configs.pop(
                    "is_pull_request", False
                )
                deployment_configs = DeploymentConfigs(**deployment_configs)

            test_run_manager.create_test_run(
                deployment=deployment,
                deployment_configs=deployment_configs,
                file_name=session.config.getoption("file_or_dir")[0],
                disable_request=disable_request,
            )
        except:
            test_run_manager.create_test_run()


def pytest_addoption(parser):
    parser.addoption(
        "--deployment",
        action="store",
        default=None,
        help="Set deployment configs",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    print("Running teardown with pytest sessionfinish...")

    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")
