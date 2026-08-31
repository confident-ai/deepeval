from pydantic import BaseModel

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.json_correctness.json_correctness import (
    JsonCorrectnessMetric,
    DEFAULT_CORRECT_REASON_NO_SCHEMA,
    DEFAULT_INCORRECT_REASON_NO_SCHEMA,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class _StubModel(DeepEvalBaseLLM):
    """Minimal judge: the schema-less mode never invokes the LLM, so this model
    only needs to exist for construction."""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-model"

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "schema-less mode must never call the LLM; got prompt: %s" % prompt
        )

    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "schema-less mode must never call the LLM; got prompt: %s" % prompt
        )


class CarSchema(BaseModel):
    make: str
    year: int
    electric: bool


def _make_metric(**kwargs) -> JsonCorrectnessMetric:
    return JsonCorrectnessMetric(model=_StubModel(), **kwargs)


def _measure(
    metric: JsonCorrectnessMetric, actual_output: str
) -> JsonCorrectnessMetric:
    test_case = LLMTestCase(
        input="Please return a JSON object.",
        actual_output=actual_output,
    )
    metric.measure(test_case)
    return metric


class TestJsonCorrectnessSchemaLess:
    """Offline, no-LLM tests for the schema-less `expected_schema=None` mode.

    These never need an API key: schema-less mode is fully deterministic and
    never invokes the LLM.
    """

    # ---- default behaviour doesn't regress (schema mode intact) ----

    def test_schema_mode_still_validates(self):
        metric = _make_metric(expected_schema=CarSchema)
        assert metric.expected_schema is CarSchema
        assert metric._validates_as_json(
            '{"make": "Tesla", "year": 2023, "electric": true}'
        )
        assert not metric._validates_as_json('{"make": "Tesla"}')

    def test_measure_with_schema_valid_scores_one(self):
        metric = _measure(
            _make_metric(expected_schema=CarSchema, async_mode=False),
            '{"make": "Tesla", "year": 2023, "electric": true}',
        )
        assert metric.score == 1

    def test_schema_mode_invalid_does_not_validate(self):
        # In schema mode the metric validates against the pydantic schema; an
        # object missing a required field is rejected. (This is the pre-existing
        # default behaviour, exercised offline against the helper.)
        assert not JsonCorrectnessMetric(
            model=_StubModel(), expected_schema=CarSchema
        )._validates_as_json('{"make": "Tesla"}')
        assert JsonCorrectnessMetric(
            model=_StubModel(), expected_schema=CarSchema
        )._validates_as_json(
            '{"make": "Tesla", "year": 2023, "electric": true}'
        )

    # ---- new capability: schema-less mode ----

    def test_schemeless_valid_json_scores_one(self):
        metric = _measure(
            _make_metric(async_mode=False),
            '{"make": "Tesla", "year": 2023, "electric": true}',
        )
        assert metric.score == 1
        assert metric.reason == DEFAULT_CORRECT_REASON_NO_SCHEMA
        assert metric.success is True

    def test_schemeless_invalid_json_scores_zero(self):
        metric = _measure(
            _make_metric(async_mode=False),
            '{"make": "Tesla"',
        )
        assert metric.score == 0
        assert metric.reason == DEFAULT_INCORRECT_REASON_NO_SCHEMA
        assert metric.success is False

    def test_schemeless_empty_string_fails(self):
        # Empty actual_output is rejected by the shared param validation before
        # the metric runs, so an empty string can never be scored as valid JSON.
        metric = _make_metric(async_mode=False)
        with pytest.raises(MissingTestCaseParamsError):
            _measure(metric, "")
        assert metric.score is None

    def test_schemeless_whitespace_only_fails(self):
        metric = _measure(_make_metric(async_mode=False), "   \n\t ")
        assert metric.score == 0

    def test_schemeless_non_json_text_fails(self):
        metric = _measure(_make_metric(async_mode=False), "not json at all")
        assert metric.score == 0

    def test_schemeless_parseable_primitives_accepted(self):
        # Any well-formed JSON value is valid, not just objects.
        for raw in ['"a string"', "123", "true", "null", "[1, 2, 3]"]:
            metric = _measure(_make_metric(async_mode=False), raw)
            assert metric.score == 1, raw

    def test_schemeless_include_reason_false(self):
        metric = _measure(
            _make_metric(async_mode=False, include_reason=False),
            '{"a": 1}',
        )
        assert metric.score == 1
        assert metric.reason is None

    def test_schemeless_async_mode(self):
        metric = _measure(
            _make_metric(async_mode=True),
            '{"a": 1}',
        )
        assert metric.score == 1
        assert metric.reason == DEFAULT_CORRECT_REASON_NO_SCHEMA

    def test_schemeless_async_invalid(self):
        metric = _measure(_make_metric(async_mode=True), "{bad")
        assert metric.score == 0
        assert metric.reason == DEFAULT_INCORRECT_REASON_NO_SCHEMA

    def test_verbose_logs_built(self):
        metric = _measure(_make_metric(async_mode=False), '{"a": 1}')
        assert metric.verbose_logs is not None
