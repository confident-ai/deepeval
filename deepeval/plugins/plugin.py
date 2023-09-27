import pytest
import os
from rich import print
from deepeval.api import Api, TestRun
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_ENV_VAR, PYTEST_RUN_TEST_NAME


def pytest_sessionstart(session: pytest.Session):
    global test_filename
    test_run = TestRun(
        testFile=session.config.getoption("file_or_dir")[0],
        testCases=[],
        metricScores=[],
        configurations={},
    )
    test_filename = test_run.save()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None  # continue with the default protocol


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    # Code before yield will run before the test teardown

    # yield control back to pytest for the actual teardown
    yield

    # Code after yield will run after the test teardown
    if os.getenv(PYTEST_RUN_ENV_VAR) and os.path.exists(".deepeval"):
        api: Api = Api()
        test_run = TestRun.load(test_filename)
        result = api.post_test_run(test_run)
        link = f"https://app.confident-ai.com/project/{result.projectId}/unit-tests/{result.testRunId}"
        print(
            "✅ Tests finished! View results on " f"[link={link}]{link}[/link]"
        )
        os.remove(test_filename)
