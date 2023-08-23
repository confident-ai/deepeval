# 👩‍⚖️ DeepEval

[![](https://dcbadge.vercel.app/api/server/a3K9c8GRGt)](https://discord.gg/a3K9c8GRGt)

DeepEval provides a Pythonic way to run offline evaluations on your LLM pipelines so you can launch comfortably into production. The guiding philosophy is a "Pytest for LLM" that aims to make productionizing and evaluating LLMs as easy as ensuring all tests pass.

Looking for DeepEval API? Please join the waitlist here: https://forms.gle/y3uqNBkmfxVYLXGq6

# Documentation

We highly recommend getting started through our documentation here: https://docs.confident-ai.com/docs/

Join our discord: https://discord.gg/a3K9c8GRGt

## Why DeepEval?

Deepeval aims to make writing tests for LLM applications (such as RAG) as easy as writing Python unit tests.

For any Python developer building production-grade apps, it is common to set up PyTest as the default testing suite as it provides a clean interface to quickly write tests.

However, it is often uncommon for many machine learning engineers as their feedback is often in the form of an evaluation loss.

With the advent of agents, LLMs and AI, there is yet to be a tool that can provide software-like tooling and abstractions for machine learning engineers where the feedback loop of these iterations can be significantly reduced.

It is therefore important then to build a new type of testing framework for LLMs to ensure engineers can keep iterating on their prompts, agents and LLMs while being able to continuously add to their test suite.

Introducing DeepEval.

# Installation

```
pip install deepeval
```

# QuickStart

## Individual Test Cases

Grab your API key from [https://app.confident-ai.com](https://app.confident-ai.com) to start logging!

```python
# test_example.py
import os
from deepeval.test_utils import assert_

# Optional - if you want an amazing dashboard!
os.environ["CONFIDENT_AI_API_KEY"] = "XXX"

def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output

def test_llm_output(self):
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    assert_llm_output(output, expected_output, metric="entailment")
    assert_llm_output(output, expected_output, metric="exact")
```

Once you have set that up, you can simply call pytest

```bash
python -m pytest test_example.py

# Output
Running tests ... ✅
```

Once you have ran tests, you should be able to see your dashboard on [https://app.confident-ai.com](https://app.confident-ai.com)

## Setting up metrics

### Setting up custom metrics

To define a custom metric, you simply need to define the `measure` and `is_successful` property.

```python
from deepeval.metric import Metric
class CustomMetric(Metric):
    def measure(self, a, b):
        self.success = a > b
        return 0.1

    def is_successful(self):
        return self.success

metric = CustomMetric()
```

## Integrate tightly with LangChain

We integrate DeepEval tightly with common frameworks such as Langchain and lLamaIndex.

# Synthetic Query Generation

![Synthetic Queries](assets/synthetic-query-generation.png)

Generating synthetic queries allows you to quickly evaluate the queries related to your prompts.
We help developers get up and running with a lot of example queries.

# Dashboard

Set up a simple dashboard in just 1 line of code. You can read more about how to do this [here in our documentation](https://docs.confident-ai.com/docs/quickstart/dashboard-app).

![docs/assets/dashboard-app.png](docs/assets/dashboard-screenshot.png)

# RoadMap

- [ ] Integration with LlamaIndex
- [ ] Project View To Web UI

# Authors

Built by the Confident AI Team. For any questions/business enquiries - please contact jacky@twilix.io.
