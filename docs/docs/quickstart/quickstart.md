# QuickStart

Once you have installed, run the login command. During this step, you will be asked to visit https://app.confident-ai.com to grab your API key.

Note: this step is entirely optional if you do not wish to track your results but we highly recommend it so you can view how results differ over time.

```bash
deepeval login

# If you already have an API key and want to feed it in through CLI
deepeval login --api-key $API_KEY
```

Once you have logged in, you can generate a sample test file as shown below. This test file allows you to quickly get started modifying it with various tests. (More on this later)

```bash
deepeval test generate --output-file test_sample.py
```

Once you have generated the test file, you can then run tests as shown.

```bash
deepeval test run test_sample.py
# if you wish to fail first 
deepeval test run -x test_sample.py
# If you want to run an interactive debugger when a test fails
deepeval test run --pdb test_sample.py
```

## Diving Into The Example

Diving into the example, it shows what a sample test looks like. It uses `assert_overall_score` to ensure that the overall score exceeds a certain threshold. We recommend experimenting with different tests to ensure that the LLMs work as intended across domains such as Bias, Answer Relevancy and Factual Consistency.

With overall score, if you leave out `query` or `expected_output`, DeepEval will automatically run the relevant tests.

```python
from deepeval.metrics.overall_score import assert_overall_score


def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    assert_overall_score(query, output, expected_output, context)
```

