import pytest
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
    api: Api = Api()
    test_run = TestRun.load(test_filename)
    api.post_test_run(test_run)
