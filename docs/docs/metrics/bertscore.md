# BertScore

BertScore is a common metric for measuring the similarity between 2 texts. Under the hood, it uses embedding models to measure the similarity between an input and output text. There are, however, a few known issues with this approach: 
- BertScore fails to know when 2 texts are similar but contradicting (for example - there is a high similarity between "This is awesome!" and "This is not awesome!").
- BertScore also fails to provide high similarity when 2 texts are of varying length. New research into asymmetric embedding models have attempted to resolve this issue but continues to remain a tough problem to solve.

These issues, however, are largely solved by coupling similarity comparisons with entailment scores.

## Usage

```python
from deepeval.test_utils import assert_llm_output
from deepeval.metrics.bertscore_metric import BertScoreMetric

def generate_llm_output(text_input: str):
    return text_input


def test_llm_output(self):
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    metric = BertScoreMetric()
    assert_llm_output(output, expected_output, metric=metric)

```

### Arguments

- `minimum_score` - The minimum score required for this to be considered a successful metric. The higher the better.
- `model_name` - The name of the model to use. Defaults to `sentence-transformers/all-mpnet-base-v2`
