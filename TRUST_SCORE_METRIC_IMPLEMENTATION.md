# TrustScoreMetric Implementation

## What is solved?
This PR resolves the feature request to add a new evaluation metric called `TrustScoreMetric` to the DeepEval library (Issue #2586). The new metric evaluates the trustworthiness of an LLM's output based on the sources it retrieved during RAG retrieval. This serves as an orthogonal metric to faithfulness, differentiating between highly trusted sources (like SEC filings) and less trusted sources (like unverified blog posts).

## How is it solved?
1. **Core Implementation**: Added `TrustScoreMetric` to `deepeval/metrics/trust_score/trust_score.py`, inheriting from `BaseMetric`.
2. **Parameters**:
   - `source_tiers`: A dictionary mapping source substrings to a tier (1 to 5).
   - `threshold`: A score threshold to determine if the metric is successful (defaults to 0.7).
3. **Scoring Logic**:
   - Parses the `retrieval_context` list in the `LLMTestCase`.
   - Iterates over each context chunk and searches for substring matches from the `source_tiers` dictionary keys.
   - Assigns a score per matched source tier:
     - T1 = 1.0
     - T2 = 0.8
     - T3 = 0.6
     - T4 = 0.4
     - T5 = 0.2
   - Unmatched sources receive a default score of 0.5.
   - The final score is the average of the chunk scores.
4. **Reasoning**: Automatically builds a human-readable `reason` showing exactly which source matched which tier.
5. **Exports**: Exported the new metric in `deepeval/metrics/trust_score/__init__.py` and exposed it in the top-level `deepeval/metrics/__init__.py`.

## How to verify it?
There are two ways to verify the functionality:

**1. Run the test suite**
We have written comprehensive automated tests covering high trust, low trust, mixed trust, unmatched trust, and edge cases (like empty retrieval context).
```bash
poetry install
poetry run pytest tests/test_metrics/test_trust_score_metric.py
```

**2. Run the provided minimal example**
A self-contained usage example script was added to the examples directory. This runs two assertions (one passing for a highly trusted source, one failing for a low trusted source).
```bash
poetry run pytest examples/getting_started/test_trust_score.py
```
