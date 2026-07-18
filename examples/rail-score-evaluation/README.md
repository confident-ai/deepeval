# RAIL Score Evaluation

Evaluate LLM outputs across 8 responsible AI dimensions using [RAIL Score](https://responsibleailabs.ai/) as a custom DeepEval metric.

**Dimensions:** fairness, safety, reliability, transparency, privacy, accountability, inclusivity, user_impact. Each scored 0-10 by the API, normalized to 0-1 for DeepEval.

## Setup

```bash
pip install -r requirements.txt
export RAIL_API_KEY="rail_..."  # Free tier at https://responsibleailabs.ai
```

## Usage

### As a standalone metric

```python
from deepeval.test_case import LLMTestCase
from rail_score_metric import RAILScoreMetric

metric = RAILScoreMetric(threshold=0.5, mode="basic")
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

metric.measure(test_case)
print(f"Score: {metric.score:.2f}")           # 0-1 overall
print(f"Breakdown: {metric.score_breakdown}")  # per-dimension dict
print(f"Reason: {metric.reason}")
```

### With evaluate()

```python
from deepeval import evaluate

results = evaluate(
    test_cases=[test_case],
    metrics=[RAILScoreMetric(threshold=0.5)],
)
```

### Deep mode with domain context

```python
metric = RAILScoreMetric(
    threshold=0.6,
    mode="deep",         # Detailed per-dimension explanations
    domain="healthcare",  # Domain-specific scoring
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Minimum score (0-1) to pass |
| `mode` | `"basic"` | `"basic"` (fast) or `"deep"` (with explanations) |
| `domain` | `"general"` | `"general"`, `"healthcare"`, `"finance"`, `"legal"`, `"education"`, `"code"` |
| `dimensions` | all 8 | Subset of dimensions to evaluate |
| `strict_mode` | `False` | Binary 0/1 scoring |
| `async_mode` | `True` | Async evaluation (DeepEval default) |

## Run the example

```bash
python example_evaluation.py
```

## Links

- [RAIL Score SDK on PyPI](https://pypi.org/project/rail-score-sdk/)
- [SDK Documentation](https://docs.responsibleailabs.ai)
- [API Reference](https://docs.responsibleailabs.ai/api-reference/evaluation)
