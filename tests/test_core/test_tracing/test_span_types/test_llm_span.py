from deepeval.tracing import observe, update_llm_span
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="llm", model="gpt-4")
def llm_generation(prompt: str) -> str:
    output = f"Generated response to: {prompt}"
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=len(output.split()),
    )
    return output


@observe(
    type="llm",
    model="gpt-4-turbo",
    cost_per_input_token=0.01,
    cost_per_output_token=0.03,
)
def llm_with_costs(prompt: str) -> str:
    output = f"Premium response: {prompt}"
    update_llm_span(
        input_token_count=10,
        output_token_count=20,
    )
    return output


@observe(type="llm")
def llm_minimal(prompt: str) -> str:
    return f"Response: {prompt}"


@observe(type="llm", model="gpt-4", name="custom_llm_name")
def llm_with_custom_name(prompt: str) -> str:
    return f"Named LLM: {prompt}"


@observe(type="llm", model="gpt-3.5-turbo")
def llm_with_full_attributes(prompt: str) -> str:
    output = "Full attributes response"
    update_llm_span(
        model="gpt-4",  # Override model at runtime
        input_token_count=100,
        output_token_count=500,
        cost_per_input_token=0.03,
        cost_per_output_token=0.06,
    )
    return output


@observe(type="llm")
def llm_set_model_at_runtime(prompt: str) -> str:
    output = f"Generated: {prompt}"
    update_llm_span(
        model="claude-3-opus",
        input_token_count=20,
        output_token_count=40,
    )
    return output


class TestLlmSpan:

    @trace_test("span_types/llm_span_schema.json")
    def test_llm_generation(self):
        llm_generation("Hello world")

    @trace_test("span_types/llm_with_costs_schema.json")
    def test_llm_with_costs(self):
        llm_with_costs("Premium query")

    @trace_test("span_types/llm_minimal_schema.json")
    def test_llm_minimal(self):
        llm_minimal("Simple prompt")

    @trace_test("span_types/llm_custom_name_schema.json")
    def test_llm_with_custom_name(self):
        llm_with_custom_name("Test")

    @trace_test("span_types/llm_full_attributes_schema.json")
    def test_llm_full_attributes(self):
        llm_with_full_attributes("Analyze this")

    @trace_test("span_types/llm_runtime_model_schema.json")
    def test_llm_set_model_at_runtime(self):
        llm_set_model_at_runtime("Hello")
