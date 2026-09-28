# Judge I/O Capture Implementation

## Summary

This implementation adds the ability to capture and retain the raw prompts sent to the judge LLM and the raw responses received from it, solving the problem where the framework's metric path retains the extracted verdict and reason but discards the judge's raw input and output.

## Changes Made

### 1. Base Metric Classes (`deepeval/metrics/base_metric.py`)

Added two new fields to `BaseMetric`, `BaseConversationalMetric`, and `BaseArenaMetric`:

```python
judge_prompts: Optional[List[str]] = None
judge_responses: Optional[List[str]] = None
```

Added a helper method `_record_judge_call(prompt, response)` to each base class that:
- Initializes the lists on first call
- Appends the stringified prompt and response to the lists
- Tracks all judge LLM calls made during metric evaluation

### 2. Utility Functions (`deepeval/metrics/utils.py`)

Modified `generate_with_schema_and_extract()` and `a_generate_with_schema_and_extract()` to call `metric._record_judge_call(prompt, result)` after receiving the LLM response but before extracting the structured data.

This captures:
- The raw prompt sent to the LLM
- The raw response (either a Pydantic model or JSON string)

### 3. GEval Direct Calls (`deepeval/metrics/g_eval/g_eval.py`)

Updated `_evaluate()` and `_a_evaluate()` methods to capture judge I/O when using `generate_raw_response()` directly (the logprob-weighted scoring path).

Captures `res.choices[0].message.content` as the raw response.

### 4. Conversational GEval (`deepeval/metrics/conversational_g_eval/conversational_g_eval.py`)

Applied the same fix as GEval for the conversational variant's `evaluate()` and `a_evaluate()` methods.

### 5. Goal Accuracy (`deepeval/metrics/goal_accuracy/goal_accuracy.py`)

Updated `_generate_reason()` and `_a_generate_reason()` which call `model.generate()` directly (bypassing the utility function).

### 6. MetricData Schema (`deepeval/tracing/api.py`)

Added two new fields to the `MetricData` Pydantic model:

```python
judge_prompts: Optional[List[str]] = Field(None, alias="judgePrompts")
judge_responses: Optional[List[str]] = Field(None, alias="judgeResponses")
```

These fields use camelCase aliases for API compatibility.

### 7. Metric Data Serialization (`deepeval/evaluate/utils.py`)

Updated `create_metric_data()` and `create_arena_metric_data()` to forward the new fields from the metric instance to the `MetricData` object.

## Usage

After running a metric evaluation, you can access the captured judge I/O:

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

metric.measure(test_case)

# Access the captured judge I/O
print("Prompts sent to judge LLM:")
for i, prompt in enumerate(metric.judge_prompts):
    print(f"\n--- Prompt {i+1} ---")
    print(prompt)

print("\n\nResponses from judge LLM:")
for i, response in enumerate(metric.judge_responses):
    print(f"\n--- Response {i+1} ---")
    print(response)
```

## Benefits

1. **Auditability**: You can now verify exactly what was asked to the judge LLM and what it responded with
2. **Debugging**: When metrics produce unexpected scores, you can inspect the raw prompts and responses to understand why
3. **Custom Analysis**: The raw data enables custom post-processing or analysis of judge behavior
4. **No Breaking Changes**: The fields are optional and default to `None`, so existing code continues to work

## Coverage

The implementation captures judge I/O for:

- All metrics using `generate_with_schema_and_extract()` (most metrics)
- GEval metrics using `generate_raw_response()` for logprob-weighted scoring
- Conversational GEval metrics
- Goal Accuracy metric (which calls `model.generate()` directly)

## Testing

All modified files pass Python syntax validation. The implementation:

- Does not break existing functionality (backward compatible)
- Adds minimal overhead (string conversion of prompts/responses)
- Works with both sync and async evaluation paths
- Handles both native and non-native model paths

## Replayability extension

The first version made judge prompts *inspectable* but not *replayable*: for
structured-output calls `judge_responses` held a Python rendering of the parsed
object, GEval logprobs were dropped, no provider metadata was kept, and cache
hits returned the earlier run's captures with nothing to say no judge was called.
The extension closes those four gaps.

### What is captured now

- `judge_responses[i]` is the assistant message content taken from the
  captured response body whenever the provider exchange was captured (OpenAI
  models). For other models it falls back to `model_dump_json()` for pydantic
  results, else `str()`.
- `judge_exchanges[i]` (index-aligned with `judge_prompts`, `None` when the
  model does not expose the raw response) holds:
  `status` (`live`/`replay`), `provider`, `request_model`, `served_model`,
  `request_id` (`x-request-id`), `response_id`, `system_fingerprint`,
  `finish_reason`, `usage`, `response_content`, `logprobs`, `http_status`,
  `captured_at`, `discarded_attempts` (attempts that reached the provider but
  were not kept), and the response body in two forms:
  - `raw_response_body`: the HTTP entity body (after any `Content-Encoding` is
    undone) decoded to text with the response charset. This matches the pilot
    ledger's `response_body`; it is text, not a byte-level record.
  - `raw_response_body_base64` and `raw_response_body_sha256`: the entity
    body's bytes and their SHA-256, for byte-level fidelity checks.
- `judge_capture_status` on the metric and on `MetricData`: `live` when this
  run made the judge calls, `replay` when the result came from the cache.

All three are forwarded into `MetricData` as `judgeExchanges` and
`judgeCaptureStatus`.

### How

- `deepeval/judge_capture.py` holds a context-var capture scope. Metrics open
  it around each judge call; `OpenAIModel` sees the scope and goes through the
  SDK's `with_raw_response` mode, recording the body *before* the SDK parses
  it. Outside a scope (direct `model.generate()` calls) the model behaves
  exactly as before.
- Every metric class's public `measure()` / `a_measure()` is wrapped at class
  creation to reset the capture lists when the outermost call starts. A second
  `measure()` on the same instance, direct or through `evaluate()`, starts from
  empty; `measure()` delegating to `a_measure()` keeps one run together.
- `Cache.get_metric_data` returns cache hits marked `replay` (status on
  `MetricData` and on every exchange, `evaluation_cost` 0); the async and
  sync evaluation paths also load those replay-marked captures onto the metric.
- Opt-in `DEEPEVAL_JUDGE_PROVENANCE=1` adds a `template_fingerprint` to the
  cache key. It hashes:
  - the deepeval version;
  - the template files: the runtime bundle `deepeval/templates` (including
    the compiled `templates.json`), the metric's package directory, and the
    file defining a custom `evaluation_template`;
  - the metric's configuration: every constructor argument stored on the
    instance (for GEval: `criteria`, `rubric`, `top_logprobs`, `strict_mode`,
    `evaluation_params`, the judge model's name, temperature and generation
    kwargs, ...), with the selected `evaluation_template` identified by the
    name and source of every class in its MRO. Two template classes in the
    same file therefore fingerprint differently.

  Values the metric derives during `measure()` are left out (GEval's generated
  `evaluation_steps`, which the existing cache check already compares, and
  TaskCompletion's inferred `task`; an explicitly given `task` is included),
  so a fresh and a used instance agree. The fingerprint errs towards misses:
  values it cannot represent contribute their type only.
- Without the flag the rubric is not part of the cache key (upstream
  behaviour): changing it returns the earlier result, now marked `replay`.

### Recomputing a GEval score from the archive

```python
import json
from deepeval.judge_capture import rebuild_chat_completion
from deepeval.metrics.g_eval.utils import calculate_weighted_summed_score

exchange = metric_data.judge_exchanges[-1]
raw = json.loads(exchange["response_content"])["score"]
weighted = calculate_weighted_summed_score(raw, rebuild_chat_completion(exchange))
low, high = score_range
assert metric_data.score == (weighted - low) / (high - low)
```

### Note on template mutation

Prompts are rendered from the compiled `deepeval/templates/metrics/templates.json`
(built by `scripts/compile_metric_templates.py`), not from the per-metric
`templates/*.txt` sources. Editing a `.txt` file in an installed package does
not change any prompt a live call sends. The resolver also caches the loaded
bundle and compiled Jinja templates per process, so an edit to
`templates.json` during a run takes effect only after
`deepeval.templates.resolver.clear_metric_template_cache()`. A custom
`evaluation_template` class is the cleaner way to vary a rubric.

### Tests

`tests/test_core/test_judge_capture.py` (no network; real `OpenAIModel` over
an `httpx.MockTransport`). Includes end-to-end `evaluate()` runs showing a
cache miss and a changed prompt after a rubric change, a `top_logprobs` change
and a switch between two template classes, and repeated direct measurement.

## Files Modified

1. `deepeval/metrics/base_metric.py` - Added fields and helper method
2. `deepeval/metrics/utils.py` - Added capture calls to utility functions
3. `deepeval/metrics/g_eval/g_eval.py` - Added capture for direct model calls
4. `deepeval/metrics/conversational_g_eval/conversational_g_eval.py` - Added capture for direct model calls
5. `deepeval/metrics/goal_accuracy/goal_accuracy.py` - Added capture for direct model calls
6. `deepeval/tracing/api.py` - Added fields to MetricData schema
7. `deepeval/evaluate/utils.py` - Forwarded fields in serialization functions
