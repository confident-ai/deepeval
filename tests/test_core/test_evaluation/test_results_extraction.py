from types import SimpleNamespace
from deepeval.evaluate.utils import extract_span_test_results
from deepeval.tracing.api import TraceSpanApiStatus


def test_extract_span_result_success_with_enum_status():
    span_enum = SimpleNamespace(
        name="span",
        status=TraceSpanApiStatus.SUCCESS,
        metrics_data=[
            SimpleNamespace(
                name="m",
                success=True,
                score=1,
                threshold=None,
                strict_mode=False,
                evaluation_model=None,
                error=None,
                evaluationCost=None,
                verboseLogs=None,
            )
        ],
        input=None,
        output=None,
        expected_output=None,
        context=None,
        retrieval_context=None,
    )

    res = extract_span_test_results(span_enum)[0]
    assert res.success is True
