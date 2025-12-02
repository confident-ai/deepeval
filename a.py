from deepeval.metrics import *
from deepeval.test_case import (
    LLMTestCase,
    MLLMImage,
    ConversationalTestCase,
    Turn,
    LLMTestCaseParams,
)
from deepeval.dataset import Golden, ConversationalGolden
from deepeval import evaluate

image = MLLMImage(local=True, url="./car_drift.png")
image2 = MLLMImage(local=True, url="./car_drift1.jpg")
img = MLLMImage(local=True, url="./1.png")

test_case = LLMTestCase(
    input="Tell me about some cars",
    actual_output=f"There's middle-class bmw cars {image} and luxury sports like lambos {image2}.",
    expected_output=f"Best car for middle-class family is a bmw {image}",
    retrieval_context=[
        f"There's 2 types of cars: ",
        f"Middle-class sports the BMW cars: {image}",
        f"Luxury sports, the Lmaborghini cars {image2}",
    ],
)

metric = MultimodalContextualRecallMetric()

evaluate([test_case], [metric])
