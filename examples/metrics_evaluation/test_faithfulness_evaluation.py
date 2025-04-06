# Example: Using MovieKGFaithfulnessTemplate with DeepEval
# pip install deepeval openai datasets

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from examples.metrics_evaluation.templates import MovieKGFaithfulnessTemplate  

# Mock: Your model's output and the reference context (e.g., from Neo4j movie graph)
question = "Did Clint Eastwood direct and act in the movie 'Unforgiven'?"
actual_output = "Yes, Clint Eastwood was both the director and lead actor in the 1992 film 'Unforgiven'."
retrieval_context = [
    "Movie: Unforgiven",
    "Year: 1992",
    "Director: Clint Eastwood",
    "Actors: Clint Eastwood, Gene Hackman"
]

metric = FaithfulnessMetric(
    threshold=0.9,
    model="gpt-3.5-turbo",
    include_reason=True,
    evaluation_template=MovieKGFaithfulnessTemplate
)

# Create a test case for evaluation
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])