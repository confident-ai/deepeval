import pytest
import asyncio
import os

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.tracing import observe, update_current_trace


pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="needs OPENAI_API_KEY",
)


@observe()
def llm_app(input: str) -> str:
    mock_output = f"I can't answer that question: {input}"

    update_current_trace(input=input, output=mock_output)
    return mock_output


@observe()
async def a_llm_app(input: str) -> str:
    mock_output = f"I can't answer that question: {input}"
    update_current_trace(input=input, output=mock_output)
    return mock_output


class TestEvaluate:

    def test_skip_on_missing_params(self):
        error_config = ErrorConfig(skip_on_missing_params=True)
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
        )
        evaluation_result = evaluate(
            test_cases=[test_case],
            metrics=[FaithfulnessMetric()],
            error_config=error_config,
        )
        assert evaluation_result.test_results[0].success
        assert len(evaluation_result.test_results) == 1

        async_config = AsyncConfig(run_async=False)
        evaluation_result = evaluate(
            test_cases=[test_case],
            metrics=[FaithfulnessMetric()],
            error_config=error_config,
            async_config=async_config,
        )

        assert len(evaluation_result.test_results) == 1
        assert evaluation_result.test_results[0].success

    def test_error_on_missing_params(self):
        error_config = ErrorConfig(skip_on_missing_params=False)
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
        )
        with pytest.raises(MissingTestCaseParamsError):
            evaluate(
                test_cases=[test_case],
                metrics=[FaithfulnessMetric()],
                error_config=error_config,
            )

        async_config = AsyncConfig(run_async=False)
        with pytest.raises(MissingTestCaseParamsError):
            evaluate(
                test_cases=[test_case],
                metrics=[FaithfulnessMetric()],
                error_config=error_config,
                async_config=async_config,
            )


class TestEvalsIterator:

    def test_async_evals_iterator(self):
        goldens = [
            Golden(
                input="What is the capital of France?",
                retrieval_context=["France is the capital of France"],
            ),
            Golden(
                input="What is the capital of Germany?",
            ),
        ]
        dataset = EvaluationDataset(goldens=goldens)
        for golden in dataset.evals_iterator(
            metrics=[AnswerRelevancyMetric()],
            async_config=AsyncConfig(run_async=True),
        ):
            task = asyncio.create_task(a_llm_app(golden.input))
            dataset.evaluate(task)
        assert True

    def test_evals_iterator(self):
        goldens = [
            Golden(
                input="What is the capital of France?",
                retrieval_context=["France is the capital of France"],
            ),
            Golden(
                input="What is the capital of Germany?",
            ),
        ]

        dataset = EvaluationDataset(goldens=goldens)
        for golden in dataset.evals_iterator(
            metrics=[AnswerRelevancyMetric()],
            async_config=AsyncConfig(run_async=False),
        ):
            llm_app(golden.input)

        assert True

    def test_skip_on_missing_params(self):
        goldens = [
            Golden(
                input="What is the capital of France?",
                retrieval_context=["France is the capital of France"],
            ),
            Golden(
                input="What is the capital of Germany?",
            ),
        ]

        dataset = EvaluationDataset(goldens=goldens)
        for golden in dataset.evals_iterator(
            metrics=[FaithfulnessMetric()],
            error_config=ErrorConfig(skip_on_missing_params=True),
        ):
            llm_app(golden.input)

        assert True

    def test_error_on_missing_params(self):
        goldens = [
            Golden(
                input="What is the capital of France?",
                retrieval_context=["France is the capital of France"],
            ),
            Golden(
                input="What is the capital of Germany?",
            ),
        ]

        dataset = EvaluationDataset(goldens=goldens)

        with pytest.raises(MissingTestCaseParamsError):
            for golden in dataset.evals_iterator(
                metrics=[FaithfulnessMetric()],
                error_config=ErrorConfig(skip_on_missing_params=False),
            ):
                llm_app(golden.input)
