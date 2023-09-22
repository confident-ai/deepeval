import pytest
import os
from deepeval.api import Api, TestRun

from deepeval.constants import PYTEST_RUN_ENV_VAR


def pytest_sessionstart(session):
    global test_filename
    test_run = TestRun(
        testFile="-",
        testCases=[],
        metricScores=[],
        configurations={},
    )
    test_filename = test_run.save()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    # Code before yield will run before the test teardown
    api: Api = Api()

    # yield control back to pytest for the actual teardown
    yield

    # Code after yield will run after the test teardown
    if os.getenv(PYTEST_RUN_ENV_VAR):
        test_run = TestRun.load(test_filename)
        result = api.post_test_run(test_run)
        run_id = result["id"]
        print(
            "✅ Tests finished! View results on "
            + f"https://app.confident-ai.com/unit-tests/{run_id}/test-cases"
        )
