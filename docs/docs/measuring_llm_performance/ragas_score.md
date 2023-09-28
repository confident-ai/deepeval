# RAGAS Score

RAGAS is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where ragas (RAG Assessment) comes in.

From the [RAGAS README](https://github.com/explodinggradients/ragas).

- Faithfulness: measures the information consistency of the generated answer against the given context. If any claims are made in the answer that cannot be deduced from context is penalized. It is calculated from answer and retrieved context.

- Context Relevancy: measures how relevant retrieved contexts are to the question. Ideally, the context should only contain information necessary to answer the question. The presence of redundant information in the context is penalized. It is calculated from question and retrieved context.

- Context Recall: measures the recall of the retrieved context using annotated answer as ground truth. Annotated answer is taken as proxy for ground truth context. It is calculated from ground truth and retrieved context.

- Answer Relevancy: refers to the degree to which a response directly addresses and is appropriate for a given question or context. This does not take the factuality of the answer into consideration but rather penalizes the present of redundant information or incomplete answers given a question. It is calculated from question and answer.

- Aspect Critiques: Designed to judge the submission against defined aspects like harmlessness, correctness, etc. You can also define your own aspect and validate the submission against your desired aspect. The output of aspect critiques is always binary. It is calculated from answer.

To use RAGAS with DeepEval, you can try the following:

Make sure to firstly install ragas to use their models where possible.

```bash
pip install ragas
```

You can then set up a test using RAGAS using a few lines of code below.

```python
from deepeval.metrics.ragas_metric import RagasMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas_metric import RagasMetric

def test_overall_score():
    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    metric = RagasMetric()
    assert_test(
        test_cases=[test_case],
        metrics=[metric],
    )
```


## Individual Metrics

You can call any of the following methodologies to use specific Ragas metrics. Below is an example of how you can do just that!

```python
from deepeval.metrics.ragas_metric import RagasContextualRelevancyMetric
from deepeval.metrics.ragas_metric import RagasAnswerRelevancyMetric
from deepeval.metrics.ragas_metric import RagasFaithfulnessMetric
from deepeval.metrics.ragas_metric import RagasContextRecallMetric
from deepeval.metrics.ragas_metric import RagasHarmfulnessMetric

def test_individual_metrics():
    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    metrics = [
        RagasContextualRelevancyMetric(),
        RagasAnswerRelevancyMetric(),
        RagasFaithfulnessMetric(),
        RagasContextRecallMetric(),
        RagasHarmfulnessMetric(),
    ]
    for metric in metrics:
        score = metric.measure(test_case)
        print(f"{metric.__name__}: {score}")
```

This will print the individual scores for each metric.

## Other Metrics

In addition to the individual metrics, we have also added the following metrics:

```python
from deepeval.metrics.ragas_metric import RagasConcisenessMetric
from deepeval.metrics.ragas_metric import RagasCorrectnessMetric
from deepeval.metrics.ragas_metric import RagasCoherenceMetric
from deepeval.metrics.ragas_metric import RagasMaliciousnessMetric

def test_other_metrics():
    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    metrics = [
        RagasConcisenessMetric(),
        RagasCorrectnessMetric(),
        RagasCoherenceMetric(),
        RagasMaliciousnessMetric(),
    ]
    for metric in metrics:
        score = metric.measure(test_case)
        print(f"{metric.__name__}: {score}")
```

This will print the scores for each of the other metrics.


