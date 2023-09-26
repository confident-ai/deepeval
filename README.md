<div style="text-align: center;">
    <h1>👩‍⚖️ DeepEval</h1>
</div>

<div style="width: 700px; height: 200px; overflow: hidden;">
  <img src="assets/deepevals-cover.jpeg" alt="DeepEval" style="width: 100%; height: 200px; object-fit: cover;">
</div>
<br>

<div align="center">

[![](https://dcbadge.vercel.app/api/server/a3K9c8GRGt)](https://discord.gg/a3K9c8GRGt)

[![PyPI version](https://badge.fury.io/py/deepeval.svg)](https://badge.fury.io/py/deepeval)<a target="_blank" href="https://colab.research.google.com/drive/1HxPWwNdNnq6cLkMh4NQ_pAAPgd8vlOly?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Stay%20Updated%20On%20X)](https://twitter.com/colabdog)
</div>

DeepEval provides a Pythonic way to run offline evaluations on your LLM pipelines so you can launch comfortably into production. The guiding philosophy is a "Pytest for LLM" that aims to make productionizing and evaluating LLMs as easy as ensuring all tests pass.

## 🤝 Schedule a session

Would you like to be onboarded / would like a demo / want to see about our roadmap? Feel free to book in a time on our calendly here: https://calendly.com/d/z7h-75h-6dz/confident-ai-demo

We highly recommend getting started reading our documentation here: https://docs.confident-ai.com/docs/

Join our discord: https://discord.gg/a3K9c8GRGt

# Features

- Opinionated tests for answer relevancy, factual consistency, toxicness, bias
- Web UI to view tests, implementations, comparisons
- Auto-evaluation through synthetic query-answer creation

# Installation

```
pip install deepeval
```

Watch a Youtube video on how to get started here: [Get started in under 1 minute](http://www.youtube.com/watch?v=05uoNgZpnzM)

# QuickStart

![CLI Reveal](docs/assets/deepeval-cli-reveal.png)

## Running from a command-line interface

```bash
# Optional - if you want a web UI
deepeval login
# Run the API key and implementation name
deepeval login --api-key $API_KEY --implementation-name "sample"
# Generate a sample test file
deepeval test generate --output-file test_sample.py
# Run this test
deepeval test run test_sample.py
```

```bash
deepeval test run tests/test_sample.py
```

## Individual Test Cases

Grab your API key from [https://app.confident-ai.com](https://app.confident-ai.com) to start logging!

```python
# test_example.py
import os
import openai
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test

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
    expected_output = response.choices[0].message.content
    return expected_output

def test_llm_output():
    query = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    test_case = LLMTestCase(query=query, expected_output=expected_output)
    metric = FactualConsistencyMetric()
    assert_test(test_case, metrics=[metric])

```

Once you have set that up, you can simply call pytest

```bash
deepeval test run test_example.py

# Output
Running tests ... ✅
```

Once you have ran tests, you should be able to see your dashboard on [https://app.confident-ai.com](https://app.confident-ai.com)

## Setting up metrics

### Setting up custom metrics

To define a custom metric, you simply need to define the `measure` and `is_successful` property.

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics.metric import Metric
from deepeval.run_test import assert_test

# Run this test
class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(self, minimum_length: int = 3):
        self.minimum_length = minimum_length

    def measure(self, test_case: LLMTestCase):
        # sends to server
        text = test_case.output
        score = len(text)
        self.success = bool(score > self.minimum_length)
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"

def test_length_metric():
    metric = LengthMetric()
    test_case = LLMTestCase(
        output="This is a long sentence that is more than 3 letters"
    )
    assert_test(test_case, [metric])
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

## Why DeepEval?

DeepEval was created to simplify the testing process for Language Learning Model (LLM) applications such as Retrieval Augmented Generation (RAG). Our goal is to make writing tests as simple as writing unit tests in Python.

Python developers understand the value of a tool like PyTest, a testing suite known for its clean, user-friendly interface that makes writing tests efficient and straightforward. However, when we move from traditional software development to Machine Learning (ML), this streamlined process is often missing.

In the ML domain, feedback is often provided as raw evaluation loss, which is a departure from the structured feedback typically seen in software development.

With the increasing deployment of agents, LLMs, and AI, there is a growing need for a tool that provides the same familiar abstractions and tools found in general software development to ML engineers. The goal is to enable a faster feedback loop to speed up iterative improvements.

This gap in the ML landscape highlights the need for a new type of testing framework specifically designed for LLMs. This will allow engineers to continually evolve their prompts, agents, and LLMs, while continually adding new items to their test suite.

DeepEval is a tool designed to simplify and streamline LLM testing. Our aim is to change the way we write, run, automate, and manage our LLM tests.

Welcome to DeepEval.


# Authors

Built by the Confident AI Team. For any questions/business enquiries - please contact jacky@confident-ai.com

# Citation

```
@misc{deepeval,
  author = {Jacky Wong},
  title = {DeepEval: Framework to unit test LLMS},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/confident-ai/deepeval}},
}
```

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/confident-ai/deepeval/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=confident-ai/deepeval" />
</a>
