# Write a simple test case

You can write a simple test case as simply as:

```python
# test_sample.py

# test files must start
from deepeval.test_utils import assert_llm_output

def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output

def test_llm_output():
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    assert_llm_output(output, expected_output, metric="entailment")
    # You can also track queries to enable visualizing on frontend as below:
    assert_llm_output(
        output,
        expected_output,
        metric="entailment",
        query=input
    )
```

You can then run it in CLI using this:

```bash
python -m pytest test_sample.py
```

## Diving into `assert_llm_output`

`assert_llm_output` is the same as an `assert` statement in Python and will raise an error if it if does not match the specific metric. The metrics provided by default are:

- `entailment` - Natural language inference score based on a given model (using NLI Deberta Base by default) with a minimum score for `entailment`
- `exact` - An exact text string match
- `bertscore` - A cosine similarity metric using embeddings to calculate if two texts are similar.

## Writing a custom metric

With `deepeval`, you can easily set custom metrics or customize existing metrics. We recommend reading the `Define Your Own Metric` if you are.

```python
from deepeval.metrics.BertScoreMetric import BertScoreMetric

# Changing the minimum score for similarity for this model.
metric = BertScoreMetric(minimum_score=0.3)

assert_llm_output(output, expected_output, metric=metric)
```
