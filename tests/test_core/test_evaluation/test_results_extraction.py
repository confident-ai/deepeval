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


def test_multimodal_test_result_keeps_reference_fields():
    from deepeval.evaluate.utils import create_test_result
    from deepeval.test_case import LLMTestCase, MLLMImage
    from deepeval.test_case.api import create_api_test_case

    image = MLLMImage(url="https://example.com/cat.png")
    test_case = LLMTestCase(
        input=f"What is in {image}?",
        actual_output="A cat.",
        expected_output="A grey cat.",
        context=["Cats are animals."],
        retrieval_context=["The cat is grey."],
    )

    result = create_test_result(create_api_test_case(test_case, index=0))

    assert result.multimodal is True
    assert result.expected_output == "A grey cat."
    assert result.context == ["Cats are animals."]
    assert result.retrieval_context == ["The cat is grey."]
