# Entailment

Entailment is a powerful way to evaluate the performance of large language models and how they currently perform.

Entailment models are traditionally trained off NLI (Natural Language Inference) where the NLP models predict one of: 

- Entailment
- Contradiction
- Neutral

It can therefore be useful to know how a particular text is an entailment of another text.

## Usage

```python
from deepeval.test_utils import assert_llm_output
from deepeval.metrics.entailment_metric import EntailmentScoreMetric


def generate_llm_output(text_input: str):
    return text_input


def test_llm_output(self):
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    metric = EntailmentScoreMetric(minimum_score=0.2)
    assert_llm_output(output, expected_output, metric=metric)

```

### Arguments

- `minimum_score` - The minimum score required for this to be considered a successful metric. The higher the better.
- `model_name` - The name of the model to use. Defaults to `cross-encoder/nli-deberta-base`
