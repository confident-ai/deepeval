<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">The LLM Evaluation Framework</h1>
</p>

<p align="center">
    <a href="https://discord.com/invite/a3K9c8GRGt">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/a3K9c8GRGt?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.confident-ai.com/docs/getting-started">Documentation</a> |
        <a href="#-metrics-and-features">Metrics and Features</a> |
        <a href="#-quickstart">Getting Started</a> |
        <a href="#-integrations">Integrations</a> |
        <a href="https://confident-ai.com">Confident AI</a>
    <p>
</h4>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/confident-ai/deepeval.svg?color=violet">
    </a>
    <a href="https://colab.research.google.com/drive/1PPxYEBa6eu__LquGoFFJZkhYgWVYE6kh?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/confident-ai/deepeval/blob/master/LICENSE.md">
        <img alt="License" src="https://img.shields.io/github/license/confident-ai/deepeval.svg?color=yellow">
    </a>
</p>

**DeepEval** is a simple-to-use, open-source LLM evaluation framework, for evaluating large-language model systems. It is similar to Pytest but specialized for unit testing LLM outputs. DeepEval incorporates the latest research to evaluate LLM outputs based on metrics such as G-Eval, hallucination, answer relevancy, RAGAS, etc., which uses LLMs and various other NLP models that runs **locally on your machine** for evaluation.

Whether your application is implemented via RAG or fine-tuning, LangChain or LlamaIndex, DeepEval has you covered. With it, you can easily determine the optimal hyperparameters to improve your RAG pipeline, prevent prompt drifting, or even transition from OpenAI to hosting your own Llama2 with confidence.

> Want to talk LLM evaluation? [Come join our discord.](https://discord.com/invite/a3K9c8GRGt)

<br />

# 🔥 Metrics and Features

> ‼️ You can now run DeepEval's metrics on the cloud for free directly on [Confident AI](https://confident-ai.com)'s infrastructure 🥳

- Large variety of ready-to-use LLM evaluation metrics (all with explanations) powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
  - G-Eval
  - Summarization
  - Answer Relevancy
  - Faithfulness
  - Contextual Recall
  - Contextual Precision
  - RAGAS
  - Hallucination
  - Toxicity
  - Bias
  - etc. 
- Evaluate your entire dataset in bulk in under 20 lines of Python code **in parallel**. Do this via the CLI in a Pytest-like manner, or through our `evaluate()` function.
- Create your own custom metrics that are automatically integrated with DeepEval's ecosystem by inheriting DeepEval's base metric class.
- Integrates seamlessly with **ANY** CI/CD environment.
- Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://docs.confident-ai.com/docs/benchmarks-introduction), which includes:
  - MMLU
  - HellaSwag
  - DROP
  - BIG-Bench Hard
  - TruthfulQA
  - HumanEval
  - GSM8K
- [Automatically integrated with Confident AI](https://app.confident-ai.com) for continous evaluation throughout the lifetime of your LLM (app):
  - log evaluation results and analyze metrics pass / fails
  - compare and pick the optimal hyperparameters (eg. prompt templates, chunk size, models used, etc.) based on evaluation results
  - debug evaluation results via LLM traces
  - manage evaluation test cases / datasets in one place
  - track events to identify live LLM responses in production
  - real-time evaluation in production
  - add production events to existing evaluation datasets to strength evals over time

(Note that while some metrics are for RAG, others are better for a fine-tuning use case. Make sure to consult our docs to pick the right metric.)

<br />

# 🔌 Integrations

- 🦄 LlamaIndex, to [**unit test RAG applications in CI/CD**](https://docs.confident-ai.com/docs/integrations-llamaindex)
- 🤗 Hugging Face, to [**enable real-time evaluations during LLM fine-tuning**](https://docs.confident-ai.com/docs/integrations-huggingface)

<br />

# 🚀 QuickStart

Let's pretend your LLM application is a RAG based customer support chatbot; here's how DeepEval can help test what you've built.

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
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_case():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    assert_test(test_case, [answer_relevancy_metric])
```
Set your `OPENAI_API_KEY` as an environment variable (you can also evaluate using your own custom model, for more details visit [this part of our docs](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm)):

```
export OPENAI_API_KEY="..."
```

And finally, run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

**Your test should have passed ✅** Let's breakdown what happened.

- The variable `input` mimics user input, and `actual_output` is a placeholder for your chatbot's intended output based on this query.
- The variable `retrieval_context` contains the relevant information from your knowledge base, and `AnswerRelevancyMetric(threshold=0.5)` is an out-of-the-box metric provided by DeepEval. It helps evaluate the relevancy of your LLM output based on the provided context.
- The metric score ranges from 0 - 1. The `threshold=0.5` threshold ultimately determines whether your test has passed or not.

[Read our documentation](https://docs.confident-ai.com/docs/getting-started) for more information on how to use additional metrics, create your own custom metrics, and tutorials on how to integrate with other tools like LangChain and LlamaIndex.

<br />

## Evaluating Without Pytest Integration

Alternatively, you can evaluate without Pytest, which is more suited for a notebook environment.

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output="We offer a 30-day full refund at no extra costs.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
)
evaluate([test_case], [answer_relevancy_metric])
```

## Using Standalone Metrics

DeepEval is extremely modular, making it easy for anyone to use any of our metrics. Continuing from the previous example:

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output="We offer a 30-day full refund at no extra costs.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
)

answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
# Most metrics also offer an explanation
print(answer_relevancy_metric.reason)
```

Note that some metrics are for RAG pipelines, while others are for fine-tuning. Make sure to use our docs to pick the right one for your use case.

## Evaluating a Dataset / Test Cases in Bulk

In DeepEval, a dataset is simply a collection of test cases. Here is how you can evaluate these in bulk:

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...", context=["..."])
second_test_case = LLMTestCase(input="...", actual_output="...", context=["..."])

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])
```

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

<br/>

Alternatively, although we recommend using `deepeval test run`, you can evaluate a dataset/test cases without using our Pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

# Real-time Evaluations on Confident AI

We offer a [web platform](https://app.confident-ai.com) for you to:

1. Log and view all the test results / metrics data from DeepEval's test runs.
2. Debug evaluation results via LLM traces.
3. Compare and pick the optimal hyperparameteres (prompt templates, models, chunk size, etc.).
4. Create, manage, and centralize your evaluation datasets.
5. Track events in production and augment your evaluation dataset for continous evaluation.
6. Track events in production, view evaluation results and historical insights.

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
