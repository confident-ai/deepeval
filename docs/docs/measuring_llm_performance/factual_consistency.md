# Factual Consistency

Factual consistency refers to the accuracy and reliability of information presented in a piece of text, conversation, or any form of communication. It means that the information being conveyed is true, accurate, and aligns with established facts or reality. Factual consistency is crucial because it ensures that the information being shared is reliable and trustworthy. Inaccurate or inconsistent information can lead to misunderstandings, misinformation, and loss of credibility.

## Assert Factual Consistency

DeepEval offers an opinionated method for factual consistency based on entailment score.

```python
from deepeval.metrics.factual_consistency import is_factually_consistent


assert is_factually_consistent(
    output="Sarah spent the evening at the library, engrossed in a book.",
    context="After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond."
)
```

### Parameters

Diving into the arguments for `assert_factual_consistency`:

- `output` - the LLM generated text
- `context` - the text from which the LLM is supposed to reason and derive conclusions from

## Factual Consistency As A Metric

You can also use factual consistency as a metric as shown below.

```python
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test, run_test
metric = FactualConsistencyMetric(minimum_score=0.5)
test_case = LLMTestCase(input="This is an example input", actual_output=output, context=context)

# If you want to run a test, log it and check the score
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])
```

### How It Is Measured

Factual consistency is measured using natural language inference models based on the output score of the entailment class that compare the ground truth and the context from which the ground truth is done.

Firstly, text is chopped into smaller chunks/sentences. This text and the context is then fed into the NLI model. We then take the score of the entailment as the overall factual consistency score.

For those interested in further readings about measuring factual consistency, please see https://arxiv.org/pdf/2111.09525.pdf which details an approach for SUmma C.
