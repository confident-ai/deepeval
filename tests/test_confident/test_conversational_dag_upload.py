import uuid
import pytest
from deepeval.metrics import ConversationalDAGMetric, ConversationalGEval
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
)
from deepeval.test_case import MultiTurnParams
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.confident.types import ConfidentApiError


def _fetch_all_metrics():
    api = Api()
    data, _ = api.send_request(
        method=HttpMethods.GET,
        endpoint=Endpoints.METRICS_ENDPOINT,
    )
    return data["metrics"]


def _build_dag():
    extract = ConversationalTaskNode(
        instructions="Extract",
        output_label="X",
        evaluation_params=[MultiTurnParams.CONTENT],
    )
    headings = ConversationalBinaryJudgementNode(criteria="all three?")
    order = ConversationalNonBinaryJudgementNode(criteria="order?")
    extract.add_node(headings)
    extract.add_node(order)
    headings.add_verdict(False, score=0)
    headings.add_verdict(True, then=order)
    order.add_verdict("Yes", score=10)
    order.add_verdict("No", score=0)
    return DeepAcyclicGraph(root_nodes=[extract])


class TestConversationalDAG:

    def test_conversational_dag_upload_and_fetch(self):
        metric_name = str(uuid.uuid4())

        metric = ConversationalDAGMetric(name=metric_name, dag=_build_dag())

        upload_response = metric.upload()
        metric_id = upload_response["id"]

        metrics = _fetch_all_metrics()
        created = next(m for m in metrics if m["id"] == metric_id)

        assert created is not None
        assert created["name"] == metric_name
        assert created["multiTurn"] is True
        assert created["dag"] is not None
        assert len(created["dag"]["nodes"]) == 7

        duplicate_metric = ConversationalDAGMetric(
            name=metric_name, dag=_build_dag()
        )

        with pytest.raises(ConfidentApiError):
            duplicate_metric.upload()

        pulled_metric = ConversationalDAGMetric(
            name=metric_name, dag=_build_dag()
        )
        pulled_response = pulled_metric.pull()

        assert pulled_response["id"] == metric_id
        assert pulled_metric.metric_id == metric_id
        assert len(pulled_metric.dag.root_nodes) == 1
        assert len(pulled_metric.dag.indegree) == 7
        # the diamond's shared judgement node has two parents.
        assert max(pulled_metric.dag.indegree.values()) == 2

    def test_conversational_dag_pull_rejects_non_dag_metric(self):
        metric_name = str(uuid.uuid4())

        metric = ConversationalGEval(
            name=metric_name,
            evaluation_params=[MultiTurnParams.SCENARIO],
            criteria="Multi turn G-Eval metric",
        )
        metric.upload()

        pulled_metric = ConversationalDAGMetric(
            name=metric_name, dag=_build_dag()
        )
        with pytest.raises(ValueError):
            pulled_metric.pull()
