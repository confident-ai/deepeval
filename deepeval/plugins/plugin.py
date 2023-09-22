import pytest
import os
from deepeval.api import Api, TestRun

from datetime import datetime


def pytest_sessionstart(session):
    current_time = datetime.now().strftime("%H:%M:%S")
    session_name = session.name
    global test_filename
    test_run = TestRun(
        alias=session_name,
        testFile=current_time,
        testCases=[],
        metricScores=[],
        configurations={},
    )
    test_filename = test_run.save()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    # Code before yield will run before the test teardown
    api: Api = Api()

    # yield control back to pytest for the actual teardown
    yield

    # Code after yield will run after the test teardown
    test_run = TestRun.load(test_filename)
    api.post_test_run(test_run)
    # if os.path.exists(test_filename):
    #     os.remove(test_filename)
