<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/DeepEval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <a href="https://discord.gg/a3K9c8GRGt" target="_blank">
        <img src="https://img.shields.io/static/v1?label=Discord&message=Join%20Us&color=1f2937&logo=discord&logoColor=white&labelColor=1d4ed8&style=for-the-badge" alt="Join Discord">
    </a>
</p>

<p align="center">
    <a href="https://docs.confident-ai.com/docs/getting-started" target="_blank">
        Read The Docs
    </a>
    &nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;
    <a href="https://confident-ai.com" target="_blank">
        Website
    </a>
</p>

**DeepEval** is a simple-to-use, open-source evaluation framework for LLM applications. It is similar to Pytest but specialized for unit testing LLM applications. DeepEval evaluates performance based on metrics such as hallucination, answer relevancy, RAGAS, etc., using LLMs and various other NLP models **locally on your machine**.

Whether your application is implemented via RAG or fine-tuning, LangChain or LlamaIndex, DeepEval has you covered. With it, you can easily determine the optimal hyperparameters to improve your RAG pipeline, prevent prompt drifting, or even transition from OpenAI to hosting your own Llama2 with confidence.

<br />

# Features

- Large variety of ready-to-use evaluation metrics powered by LLMs, statistical methods, or NLP models that runs **locally on your machine**:
  - Hallucination
  - Answer Relevancy
  - RAGAS
  - G-Eval
  - Toxicity
  - Bias
  - etc.
- Easily create your own custom metrics that are automatically integrated with DeepEval's ecosystem by inheriting DeepEval's base metric class.
- Evaluate your entire dataset in bulk using fewer than 20 lines of Python code **in parallel**.
- [Automatically integrated with Confident AI](https://app.confident-ai.com) for continous evaluation throughout the lifetime of your LLM (app):
  - log evaluation results and analyze metrics pass / fails
  - compare and pick the optimal hyperparameters (eg. prompt templates, chunk size, models used, etc.) based on evaluation results
  - debug evaluation results via LLM traces
  - manage evaluation test cases / datasets in one place
  - track events to identify live LLM responses in production
  - add production events to existing evaluation datasets to strength evals over time

<br />

# 🚀 Getting Started 🚀

Let's pretend your LLM application is a customer support chatbot; here's how DeepEval can help test what you've built.

## Installation

```
pip install -U deepeval
```

## Create an account (highly recommended)

Although optional, creating an account on our platform will allow you to log test results, enabling easy tracking of changes and performances over iterations. This step is optional, and you can run test cases even without logging in, but we highly recommend giving it a try.

To login, run:

```
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste it into the CLI. All test cases will automatically be logged (find more information on data privacy [here](https://docs.confident-ai.com/docs/data-privacy)).

## Writing your first test case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case using DeepEval:

```python
import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluator import assert_test

def test_case():
    input = "What if these shoes don't fit?"
    context = ["All customers are eligible for a 30 day full refund at no extra costs."]

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra costs."
    hallucination_metric = HallucinationMetric(minimum_score=0.7)
    test_case = LLMTestCase(input=input, actual_output=actual_output, context=context)
    assert_test(test_case, [hallucination_metric])
```

Run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

**Your test should have passed ✅** Let's breakdown what happened.

- The variable `input` mimics user input, and `actual_output` is a placeholder for your chatbot's intended output based on this query.
- The variable `context` contains the relevant information from your knowledge base, and `HallucinationMetric(minimum_score=0.7)` is an out-of-the-box metric provided by DeepEval. It helps you evaluate the factual accuracy of your chatbot's output based on the provided context.
- The metric score ranges from 0 - 1. The `minimum_score=0.7` threshold ultimately determines whether your test has passed or not.

[Read our documentation](https://docs.confident-ai.com/docs/getting-started) for more information on how to use additional metrics, create your own custom metrics, and tutorials on how to integrate with other tools like LangChain and LlamaIndex.

<br />

## Evaluting a Dataset / Test Cases in Bulk

In DeepEval, a dataset is simply a collection of test cases. Here is how you can evaluate things in bulk:

```python
import pytest
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluator import assert_test
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...", context=["..."])
second_test_case = LLMTestCase(input="...", actual_output="...", context=["..."])

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])
```
```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```
<br/>

Alternatively, although we recommend using `deepeval test run`, you can evaluate a dataset/test cases without using pytest:
```python
from deepeval.evaluator import evaluate
...

evaluate(dataset, [hallucination_metric])
# or
dataset.evaluate([hallucination_metric])
```

# View results on Confident AI

We offer a [free web platform](https://app.confident-ai.com) for you to:

1. Log and view all test results / metrics data from DeepEval's test runs.
2. Debug evaluation results via LLM traces
3. Compare and pick the optimal hyperparameteres (prompt templates, models, chunk size, etc.).
4. Create, manage, and centralize your evaluation datasets.
5. Track events in production and augment your evaluation dataset for continous evaluation in production.

Everything on Confident AI, including how to use Confident is available [here](https://docs.confident-ai.com/docs/confident-ai-introduction).

To begin, login from the CLI:

```bash
deepeval login
```

Follow the instructions to log in, create your account, and paste your API key into the CLI.

Now, run your test file again:

```bash
deepeval test run test_chatbot.py
```

You should see a link displayed in the CLI once the test has finished running. Paste it into your browser to view the results!

![ok](https://d2lsxfc3p6r9rv.cloudfront.net/confident-test-cases.png)

<br />

# Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

<br />

# Roadmap

Features:

- [x] Implement G-Eval
- [x] Referenceless Evaluation
- [x] Production Evaluation & Logging
- [x] Evaluation Dataset Creation

Integrations:

- [x] lLamaIndex
- [ ] langChain
- [ ] Guidance
- [ ] Guardrails
- [ ] EmbedChain

<br />

# Authors

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all enquiries.

<br />

# License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.
