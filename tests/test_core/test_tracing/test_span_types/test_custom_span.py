from deepeval.tracing import observe
from tests.test_core.test_tracing.conftest import trace_test


@observe("CustomProcessor")
def custom_processor(data: str) -> str:
    return f"Processed: {data}"


@observe("DataTransformer", name="my_transformer")
def custom_with_name(data: dict) -> dict:
    return {"transformed": data}


@observe("Validator")
def custom_validator(value: str) -> bool:
    return len(value) > 0


@observe()
def default_span(input_data: str) -> str:
    return f"Default: {input_data}"


@observe(name="explicit_name_only")
def span_with_only_name(data: str) -> str:
    return f"Named: {data}"


class TestCustomSpan:

    @trace_test("span_types/custom_processor_schema.json")
    def test_custom_processor(self):
        custom_processor("test data")

    @trace_test("span_types/custom_with_name_schema.json")
    def test_custom_with_name(self):
        custom_with_name({"key": "value"})

    @trace_test("span_types/custom_validator_schema.json")
    def test_custom_validator(self):
        custom_validator("valid")

    @trace_test("span_types/default_span_schema.json")
    def test_default_span(self):
        default_span("input")

    @trace_test("span_types/span_with_only_name_schema.json")
    def test_span_with_only_name(self):
        span_with_only_name("test")
