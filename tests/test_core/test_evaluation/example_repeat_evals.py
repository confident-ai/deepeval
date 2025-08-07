from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.evaluate import AsyncConfig, DisplayConfig

# test_case = LLMTestCase(
#     input="Summarize: 'The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.'",
#     actual_output="Paris is cold though. Pineapples are great on pizza. The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
#     retrieval_context=[
#         "The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
#         "Paris is cold though. Pineapples are great on pizza.",
#     ],
# )
# evaluate(
#     test_cases=[test_case] * 3,
#     metrics=[FaithfulnessMetric(), AnswerRelevancyMetric(repeat=2)],
#     async_config=AsyncConfig(run_async=False),
#     display_config=DisplayConfig(verbose_mode=True),
# )

def test_correctness():
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output from your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        retrieval_context=[
            "The sun provides energy for plants to grow through photosynthesis, which forms the base of most ecosystems.",
            "Paris is cold though. Pineapples are great on pizza.",
        ],
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs.",
    )
    assert_test(test_case, [FaithfulnessMetric(), AnswerRelevancyMetric()])
