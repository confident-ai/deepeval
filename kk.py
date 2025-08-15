from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric, BaseMetric
from deepeval.test_case import LLMTestCase

testset = LLMTestCase(
    input="What is the capital of France?",
    expected_output="Paris",
)


class FakeMetric1(BaseMetric):
    def __init__(self, threshold: float = 0.5, _success: bool = True):
        self.threshold = threshold
        self.success = _success

    def measure(self, test_case: LLMTestCase):
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


evaluate(
    test_cases=[testset] * 80,
    metrics=[
        FakeMetric1(),
    ],
)
