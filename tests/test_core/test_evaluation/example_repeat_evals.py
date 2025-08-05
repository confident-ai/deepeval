from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import AsyncConfig, DisplayConfig

answer_relevancy_metric = AnswerRelevancyMetric(repeat=2)
test_case = LLMTestCase(
    input="Summarize: 'The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.'",
    actual_output="Paris is cold though. Pineapples are great on pizza. The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
)
evaluate(
    test_cases=[test_case] * 3,
    metrics=[answer_relevancy_metric],
    async_config=AsyncConfig(run_async=False),
    display_config=DisplayConfig(verbose_mode=True),
)