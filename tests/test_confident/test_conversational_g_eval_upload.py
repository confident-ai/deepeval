import os
import uuid
import pytest
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import TurnParams
from deepeval.metrics.g_eval import Rubric
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.confident.types import ConfidentApiError


def _fetch_all_metrics():
    api = Api()
    data, _ = api.send_request(
        method=HttpMethods.GET,
        endpoint=Endpoints.METRICS_ENDPOINT,
    )
    return data["metrics"]


class TestConversationalGEval:

    def test_conversational_geval_upload_and_fetch(self):
        metric_name = str(uuid.uuid4())

        metric = ConversationalGEval(
            name=metric_name,
            evaluation_params=[
                TurnParams.EXPECTED_OUTCOME,
                TurnParams.RETRIEVAL_CONTEXT,
                TurnParams.SCENARIO,
                TurnParams.TOOLS_CALLED,
            ],
            criteria=(
                "Test whether the assistant responses are relevant, grounded, "
                "and aligned with the expected outcome"
            ),
            rubric=[
                Rubric(score_range=(0, 5), expected_outcome="Nice"),
                Rubric(score_range=(6, 10), expected_outcome="Not so Nice"),
            ],
        )

        upload_response = metric.upload()
        metric_id = upload_response["id"]

        metrics = _fetch_all_metrics()
        created = next(m for m in metrics if m["id"] == metric_id)

        assert created["name"] == metric_name
        assert created["criteria"] == metric.criteria
        assert created["evaluationSteps"] is None
        assert created["multiTurn"] is True

        assert created["rubric"] == [
            {"scoreRange": [0, 5], "expectedOutcome": "Nice"},
            {"scoreRange": [6, 10], "expectedOutcome": "Not so Nice"},
        ]

        assert set(created["requiredParameters"]) == {
            "content",
            "role",
            "expectedOutcome",
            "retrievalContext",
            "scenario",
            "toolsCalled",
        }

        duplicate_metric = ConversationalGEval(
            name=metric_name,
            evaluation_params=[
                TurnParams.SCENARIO,
            ],
            criteria="Test whether actual output is relevant to the input given",
        )

        with pytest.raises(ConfidentApiError):
            duplicate_metric.upload()
