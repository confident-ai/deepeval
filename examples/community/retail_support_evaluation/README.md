# Retail Support Evaluation

This example evaluates twelve inline, synthetic retail support responses with
DeepEval's `GEval`, `FaithfulnessMetric`, and `AnswerRelevancyMetric`. The cases
cover order status, returns, refund status and timing, delivery disputes,
substitutions, loyalty points, invoice copies, address changes, damaged items,
quality reports, and cancellation after dispatch.

Each case keeps its expected policy path and required or forbidden response
content as structured fields. `serialize_expected_output` converts those fields
to the plain string used by `LLMTestCase.expected_output`. A separate phrase
check demonstrates zero-cost detection of literal unsupported refunds,
replacements, cancellations, delivery promises, and account changes. It only
matches configured phrases; it does not interpret negation or detect paraphrases.

All names, order IDs, policies, and responses are synthetic. The data is a small
example, not a benchmark or production policy. It uses no customer records,
credentials, or external data files.

Run the deterministic contract checks without model credentials from the
repository root:

```bash
pytest -q tests/test_examples/test_retail_support.py
```

To run the LLM-backed evaluation from this directory, configure the model
provider required by your DeepEval setup, then run:

```bash
deepeval test run retail_support.py
```

The evaluation scores are model-dependent; the deterministic checks validate
the declared phrases and case wiring rather than exact metric scores.
