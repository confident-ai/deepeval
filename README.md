<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/DeepEval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <a href="https://confident-ai.com" target="_blank">
        Website
    </a>
   &nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;
    <a href="https://docs.confident-ai.com" target="_blank">
        Documentation
    </a>
   &nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;
    <a href="https://discord.gg/a3K9c8GRGt" target="_blank">
        Discord
    </a>
</p>

**DeepEval** is a simple-to-use, open-source evaluation framework for LLM applications. Write **"unit tests" in python** to evaluate your LLM applications on metrics such as faithfulness, accuracy, relevance, and much more. Whether your application is implementated via RAG or fine-tuning, LangChain or lLamaIndex, you can use DeepEval to evaluate your LLM application in seconds. It's an production-ready alternative to **RAGAS**.

DeepEval is designed to help you choose the optimal hyperparameters to improve your RAG pipeline, select the best prompt templates, or even transition from OpenAI to hosting your own lLama2 with confidence. With DeepEval, you'll won't be frustrated with fixing breaking changes in your LLM application ever again.

<hr />
<br />

## Getting Started 🚀

Let's pretend your LLM application is a customer support chatbot, here's how Deepeval can help test what you've built.

### Installation

```
pip install -U deepeval
```

### [Optional] Create an account

Creating an account on our platform would allow you to log test results to our platform, which allows you to easily keep track of changes and performances over iterations. This step is optional and you can run test cases even without logging in, but we highly recommend giving it a try.

To login, run:

```
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste your API key in the CLI. All test cases will automatically be logged.

### Writing your first test case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case using Deepeval:

```python
import pytest
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test

def test_case():
    input = "What if these shoes don't fit?"
    context = "All customers are eligible for a 30 day full refund at no extra costs."

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra costs."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.7)
    test_case = LLMTestCase(input=input, actual_output=actual_output, context=context)
    assert_test(test_case, [factual_consistency_metric])
```

Run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

**Your test should have passed** ✅ Let's breakdown what happened.

The variable `input` mimics a user input, and `actual_output` is a placeholder for what your chatbot's supposed to output based on this query. The variable `context` contains the relevant information from your knowledge base, and `FactualConsistencyMetric(minimum_score=0.7)` is an out-of-the-box metric provided by DeepEval for you to evaluate how factually correct your chatbot's output is based on the provided context. This metric score ranges from 0 - 1, which the `minimum_score=0.7` threshold ultimately determines if your test have passed or not.

[Read our documentation](https://docs.confident-ai.com/) for more information on how to use additional and create your own custom metric, and tutorials on how to integrate with other tools like LangChain and lLamaIndex.

<hr />
<br />

## Evaluate your test results on our platform

We offer a [web platform](https://app.confident-ai.com) for you to log and view all test results from `deepeval test run`. Our platform allows you to quickly draw insights on how your metrics are improving with each test run, and to determine the optimal parameters (such as prompt templates, models, retrieval pipeline) for your specific LLM application.

To begin, login from the CLI:

```bash
deepeval login
```

Follow the instructions to login, create your account, and paste in your API key in the CLI.

Now run your test file again:

```bash
deepeval test run test_chatbot.py
```

You should see a link being displayed in the CLI once the test has finished running. Paste it in your browser to view results!

<br />

## Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

<br />

## Authors

Built by co-founders of Confident AI.

- for enquires, contact jeffreyip@confident-ai.com

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.
