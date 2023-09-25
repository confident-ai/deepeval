import os
from deepeval.dataset import (
    create_evaluation_query_answer_pairs,
    EvaluationDataset,
)

dataset: EvaluationDataset = create_evaluation_query_answer_pairs(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    context="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
    n=3,
)
dataset.review()
