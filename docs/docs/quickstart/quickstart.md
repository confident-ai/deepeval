# Write a simple test case

You can write a simple test case as simply as:

```python
import os
import openai
from deepeval.metrics.factual_consistency import assert_factual_consistency

# Optional - if you want an amazing dashboard!
os.environ["CONFIDENT_AI_API_KEY"] = "XXX"
# Name your implementation - e.g. "LangChain Implementation"
os.environ["CONFIDENT_AI_IMP_NAME"] = "QuickStart"

import openai
openai.api_key = "sk-XXX"

# Write a sample ChatGPT function
def generate_chatgpt_output(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The customer success phone line is 1200-231-231 and the customer success state is in Austin."},
            {"role": "user", "content": query}
        ]
    )
    llm_output = response.choices[0].message.content
    return llm_output

def test_factual_consistency():
    query = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_chatgpt_output(query)
    assert_factual_consistency(output, expected_output)

test_factual_consistency()
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
