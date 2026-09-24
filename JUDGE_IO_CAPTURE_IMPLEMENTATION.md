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

## Files Modified

1. `deepeval/metrics/base_metric.py` - Added fields and helper method
2. `deepeval/metrics/utils.py` - Added capture calls to utility functions
3. `deepeval/metrics/g_eval/g_eval.py` - Added capture for direct model calls
4. `deepeval/metrics/conversational_g_eval/conversational_g_eval.py` - Added capture for direct model calls
5. `deepeval/metrics/goal_accuracy/goal_accuracy.py` - Added capture for direct model calls
6. `deepeval/tracing/api.py` - Added fields to MetricData schema
7. `deepeval/evaluate/utils.py` - Forwarded fields in serialization functions
