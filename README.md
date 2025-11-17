<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">The LLM Evaluation Framework</h1>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5917" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5917" alt="confident-ai%2Fdeepeval | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/3SEyvpgu2f?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
        <a href="#-metrics-and-features">Metrics and Features</a> |
        <a href="#-quickstart">Getting Started</a> |
        <a href="#-integrations">Integrations</a> |
        <a href="https://confident-ai.com?utm_source=GitHub">DeepEval Platform</a>
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
    <a href="https://x.com/deepeval">
        <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/deepeval?style=social&logo=x">
    </a>
</p>

<p align="center">
    <!-- Keep these links. Translations will automatically update with the README. -->
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=de">Deutsch</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=es">Español</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=fr">français</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ja">日本語</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ko">한국어</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=pt">Português</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ru">Русский</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=zh">中文</a>
</p>

**DeepEval** is a simple-to-use, open-source LLM evaluation framework, for evaluating and testing large-language model systems. It is similar to Pytest but specialized for unit testing LLM outputs. DeepEval incorporates the latest research to evaluate LLM outputs based on metrics such as G-Eval, hallucination, answer relevancy, RAGAS, etc., which uses LLMs and various other NLP models that runs **locally on your machine** for evaluation.

Whether your LLM applications are RAG pipelines, chatbots, AI agents, implemented via LangChain or LlamaIndex, DeepEval has you covered. With it, you can easily determine the optimal models, prompts, and architecture to improve your RAG pipeline, agentic workflows, prevent prompt drifting, or even transition from OpenAI to hosting your own Deepseek R1 with confidence.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br />

# 🔥 Metrics and Features

> 🥳 You can now share DeepEval's test results on the cloud directly on [Confident AI](https://confident-ai.com?utm_source=GitHub)'s infrastructure

- Supports both end-to-end and component-level LLM evaluation.
- Large variety of ready-to-use LLM evaluation metrics (all with explanations) powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
  - G-Eval
  - DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
  - **RAG metrics:**
    - Answer Relevancy
    - Faithfulness
    - Contextual Recall
    - Contextual Precision
    - Contextual Relevancy
    - RAGAS
  - **Agentic metrics:**
    - Task Completion
    - Tool Correctness
  - **Others:**
    - Hallucination
    - Summarization
    - Bias
    - Toxicity
  - **Conversational metrics:**
    - Knowledge Retention
    - Conversation Completeness
    - Conversation Relevancy
    - Role Adherence
  - etc.
- Build your own custom metrics that are automatically integrated with DeepEval's ecosystem.
- Generate synthetic datasets for evaluation.
- Integrates seamlessly with **ANY** CI/CD environment.
- [Red team your LLM application](https://deepeval.com/docs/red-teaming-introduction) for 40+ safety vulnerabilities in a few lines of code, including:
  - Toxicity
  - Bias
  - SQL Injection
  - etc., using advanced 10+ attack enhancement strategies such as prompt injections.
- Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://deepeval.com/docs/benchmarks-introduction?utm_source=GitHub), which includes:
  - MMLU
  - HellaSwag
  - DROP
  - BIG-Bench Hard
  - TruthfulQA
  - HumanEval
  - GSM8K
- [100% integrated with Confident AI](https://confident-ai.com?utm_source=GitHub) for the full evaluation lifecycle:
  - Curate/annotate evaluation datasets on the cloud
  - Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
  - Fine-tune metrics for custom results
  - Debug evaluation results via LLM traces
  - Monitor & evaluate LLM responses in product to improve datasets with real-world data
  - Repeat until perfection

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

# 🔌 Integrations

- 🦄 LlamaIndex, to [**unit test RAG applications in CI/CD**](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
- 🤗 Hugging Face, to [**enable real-time evaluations during LLM fine-tuning**](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br />

# 🚀 QuickStart

Let's pretend your LLM application is a RAG based customer support chatbot; here's how DeepEval can help test what you've built.

## Installation

Deepeval works with **Python>=3.9+**.

```
pip install -U deepeval
```

## Create an account (highly recommended)

Using the `deepeval` platform will allow you to generate sharable testing reports on the cloud. It is free, takes no additional code to setup, and we highly recommend giving it a try.

To login, run:

```
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste it into the CLI. All test cases will automatically be logged (find more information on data privacy [here](https://deepeval.com/docs/data-privacy?utm_source=GitHub)).

## Writing your first test case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case to run an **end-to-end** evaluation using DeepEval, which treats your LLM app as a black-box:

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def test_case():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    assert_test(test_case, [correctness_metric])
```

Set your `OPENAI_API_KEY` as an environment variable (you can also evaluate using your own custom model, for more details visit [this part of our docs](https://deepeval.com/docs/metrics-introduction#using-a-custom-llm?utm_source=GitHub)):

```
export OPENAI_API_KEY="..."
```

And finally, run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

**Congratulations! Your test case should have passed ✅** Let's breakdown what happened.

- The variable `input` mimics a user input, and `actual_output` is a placeholder for what your application's supposed to output based on this input.
- The variable `expected_output` represents the ideal answer for a given `input`, and [`GEval`](https://deepeval.com/docs/metrics-llm-evals) is a research-backed metric provided by `deepeval` for you to evaluate your LLM output's on any custom with human-like accuracy.
- In this example, the metric `criteria` is correctness of the `actual_output` based on the provided `expected_output`.
- All metric scores range from 0 - 1, which the `threshold=0.5` threshold ultimately determines if your test have passed or not.

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more information on more options to run end-to-end evaluation, how to use additional metrics, create your own custom metrics, and tutorials on how to integrate with other tools like LangChain and LlamaIndex.

<br />

## Evaluating Nested Components

If you wish to evaluate individual components within your LLM app, you need to run **component-level** evals - a powerful way to evaluate any component within an LLM system.

Simply trace "components" such as LLM calls, retrievers, tool calls, and agents within your LLM application using the `@observe` decorator to apply metrics on a component-level. Tracing with `deepeval` is non-instrusive (learn more [here](https://deepeval.com/docs/evaluation-llm-tracing#dont-be-worried-about-tracing)) and helps you avoid rewriting your codebase just for evals:

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval import evaluate

correctness = GEval(name="Correctness", criteria="Determine if the 'actual output' is correct based on the 'expected output'.", evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])

@observe(metrics=[correctness])
def inner_component():
    # Component can be anything from an LLM call, retrieval, agent, tool use, etc.
    update_current_span(test_case=LLMTestCase(input="...", actual_output="..."))
    return

@observe
def llm_app(input: str):
    inner_component()
    return

evaluate(observed_callback=llm_app, goldens=[Golden(input="Hi!")])
```

You can learn everything about component-level evaluations [here.](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

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
# All metrics also offer an explanation
print(answer_relevancy_metric.reason)
```

Note that some metrics are for RAG pipelines, while others are for fine-tuning. Make sure to use our docs to pick the right one for your use case.

## Evaluating a Dataset / Test Cases in Bulk

In DeepEval, a dataset is simply a collection of test cases. Here is how you can evaluate these in bulk:

```python
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

dataset = EvaluationDataset(goldens=[Golden(input="What's the weather like today?")])

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=your_llm_app(golden.input)
    )
    dataset.add_test_case(test_case)

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])
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

## A Note on Env Variables (.env / .env.local)

DeepEval auto-loads `.env.local` then `.env` from the current working directory **at import time**.
**Precedence:** process env -> `.env.local` -> `.env`.
Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

# DeepEval With Confident AI

DeepEval's cloud platform, [Confident AI](https://confident-ai.com?utm_source=Github), allows you to:

1. Curate/annotate evaluation datasets on the cloud
2. Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3. Fine-tune metrics for custom results
4. Debug evaluation results via LLM traces
5. Monitor & evaluate LLM responses in product to improve datasets with real-world data
6. Repeat until perfection

Everything on Confident AI, including how to use Confident is available [here](https://www.confident-ai.com/docs?utm_source=GitHub).

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

![Demo GIF](assets/demo.gif)

<br />

## Configuration

### Environment variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br />

# Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

<br />

# Roadmap

Features:

- [x] Integration with Confident AI
- [x] Implement G-Eval
- [x] Implement RAG metrics
- [x] Implement Conversational metrics
- [x] Evaluation Dataset Creation
- [x] Red-Teaming
- [ ] DAG custom metrics
- [ ] Guardrails

<br />

# Authors

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all enquiries.

<br />

# License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.
