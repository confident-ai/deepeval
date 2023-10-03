# Write a simple test case
Create a test file:
``` bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case using Deepeval:
``` python
import pytest
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test

def test_case():
    query = "What if these shoes don't fit?"
    context = "All customers are eligible for a 30 day full refund at no extra costs."

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra costs."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.7)
    test_case = LLMTestCase(query=query, output=actual_output, context=context)
    assert_test(test_case, [factual_consistency_metric])
```
Run `test_chatbot.py` in the CLI:
```
deepeval test run test_chatbot.py
```
**Your test should have passed** ✅ Let's breakdown what happened. 

The variable `query` mimics a user input, and `actual_output` is a placeholder for what your chatbot's supposed to output based on this query. The variable `context` contains the relevant information from your knowledge base, and `FactualConsistencyMetric(minimum_score=0.7)` is an out-of-the-box metric provided by DeepEval for you to evaluate how factually correct your chatbot's output is based on the provided context. This metric score ranges from 0 - 1, which the `minimum_score=0.7` threshold ultimately determines if your test have passed or not.

[Read our documentation](https://docs.confident-ai.com/docs/) for more information on how to use additional and create your own custom metric, and tutorials on how to integrate with other tools like LangChain and lLamaIndex.

<br />

## Evaluate your test results on the web
We offer a [web platform](https://app.confident-ai.com) for you to log and view all test results from `deepeval test run`. Our platform allows you to quickly draw insights on how your metrics are improving with each test run, and to determine the optimal parameters (such as prompt templates, models, retrieval pipeline) for your specific LLM application.

To begin, login from the CLI:
``` bash
deepeval login
```
Follow the instructions to login, create your account, and paste in your API key in the CLI. 

Now run your test file again:
``` bash
deepeval test run test_chatbot.py
```

## LLMTestCase

We can broke down how to write this test and what exactly goes into an `LLMTestCase`:

Explanation of variables:
- `query`: The input query for the ChatGPT model.
- `context`: The context or conversation history for the ChatGPT model.
- `output`: The generated output response from the ChatGPT model.
- `expected_output`: The expected output response for the given input query and context.

## Running a Test

### `run_test` 

`run_test` allows you to run a test based on the metrics provided with a given number of retries and minimum successes.

You can run it with 1 or multiple test case and metrics.
```python
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric

metric_1 = FactualConsistencyMetric()
metric_2 = AnswerRelevancyMetric()
run_test(test_case, [metric_1, metric_2])
```

### `assert_test`

`assert_test` is a wrapper on top of `run_test` and can be used for enforcing errors and to maintain similar logic as other tests.

```python
assert_test(test_case, [metric_1])
```
