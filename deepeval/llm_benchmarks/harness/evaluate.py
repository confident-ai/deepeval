from lm_eval import evaluator
from deepeval.llm_benchmarks.harness.task import HarnessTasks

# Todo:
# Bring a concept of Grouped evaluations

class HarnessEvaluate:
    def __init__(self):
        results = evaluator.simple_evaluate(
            ... 
        )
        raise NotImplementedError()
    