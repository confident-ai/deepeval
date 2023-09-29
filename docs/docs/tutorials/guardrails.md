# GuardRails

Guardrails is an open-source Python package for specifying structure and type, validating and correcting the outputs of large language models (LLMs).

Guardrails has a built-in retry mechanism that allows you to validate if DeepEval metrics are passing.

```python
from typing import Dict
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult
)
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_cases import LLMTestCase

# We are registering a new validator with the name "factual-consistency". 
# This validator will be used to check the factual consistency of the generated text.
@register_validator(name="factual-consistency", data_type="string")
def factual_consistency(value: str, metadata: Dict) -> ValidationResult:
    # The 'output' is the generated text and 'context' is the text from which the LLM is supposed to reason and derive conclusions from.
    output = metadata.get('output')
    context = metadata.get('context')
    metric = FactualConsistencyMetric()
    # We use the 'assert_factual_consistency' function from deepeval to get the factual consistency score.
    
    test_case = LLMTestCase(output=output, context=context)
    score = metric.measure(test_case)
    
    # If the score is greater than the metric's minimum score, give it a PassResult
    if score >= metric.minimum_score:
        return PassResult(metadata)
    
    # If the score is less than 0.5, we return a FailResult with an appropriate error message.
    return FailResult(
        error_message=f"Factual consistency score {score} is less than {metric.minimum_score}."
    )

```