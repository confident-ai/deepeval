"""Test test cases
"""

import pytest

from deepeval.api import Api


@pytest.fixture
def api():
    return Api()


@pytest.mark.skip(reason="skip")
def test_add_golden(api: Api):
    # Test case data
    query = "SELECT * FROM table"
    actual_output = "Result"
    metric_score = 0.9
    metric_name = "Accuracy"
    success = True
    datapoint_id = "123"
    implementation_id = "456"
    metrics_metadata = {"metadata": "value"}
    context = "Test context"

    # Call the add_test_case function
    result = api.add_golden(
        query=query,
        expected_output=actual_output,
        context=context,
    )

    # Assert the result
    assert result["id"] == "clm343jd70018uq6fj3sc7d0x", result
