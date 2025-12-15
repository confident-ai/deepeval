import os
import pytest
from deepeval.metrics import ArenaGEval
from deepeval.test_case import (
    LLMTestCase,
    MLLMImage,
    ArenaTestCase,
    LLMTestCaseParams,
    Contestant,
)
from deepeval import compare

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestArenaGEval:
    """Tests for answer relevancy metric"""

    def test_normal_sync_metric_measure(self):
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hey! how are you?",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello.",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello!!",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.winner is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

    def test_normal_async_metric_measure(self):
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hey! how are you?",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello.",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello!!",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )
        metric.measure(test_case)

        assert metric.winner is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=CAR)
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a car",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a black bmw",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="A nice car",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )
        metric.measure(test_case)

        assert metric.winner is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=CAR)
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a car",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a black bmw",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="A nice car",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.winner is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=CAR)
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a car",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a black bmw",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="A nice car",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        with pytest.raises(ValueError):
            metric = ArenaGEval(
                name="Friendly",
                criteria="Choose the winner of the more accurate contestant based on the input and actual output",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model="gpt-3.5-turbo",
            )
            metric.measure(test_case)

    def test_normal_compare_method(self):
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hey! how are you?",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello.",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"Say hello.",
                actual_output="Hello!!",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            async_mode=False,
        )

        results = compare(test_cases=[test_case], metric=metric)

        assert results is not None

    def test_multimodal_compare_method(self):
        image = MLLMImage(url=CAR)
        contestant_1 = Contestant(
            name="Version 1",
            hyperparameters={"model": "gpt-3.5-turbo"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a car",
            ),
        )

        contestant_2 = Contestant(
            name="Version 2",
            hyperparameters={"model": "gpt-4o"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="That's a black bmw",
            ),
        )

        contestant_3 = Contestant(
            name="Version 3",
            hyperparameters={"model": "gpt-4.1"},
            test_case=LLMTestCase(
                input=f"What is in the image {image}",
                actual_output="A nice car",
            ),
        )
        test_case = ArenaTestCase(
            contestants=[contestant_1, contestant_2, contestant_3]
        )
        metric = ArenaGEval(
            name="Friendly",
            criteria="Choose the winner of the more accurate contestant based on the input and actual output",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            async_mode=False,
        )

        results = compare(test_cases=[test_case], metric=metric)

        assert results is not None
