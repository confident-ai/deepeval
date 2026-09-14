import pytest
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from deepeval.metrics.community import ToolArgumentValidationMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.utils import MissingTestCaseParamsError


class SearchFlightSchema(BaseModel):
    origin: str
    destination: str
    passengers: int = Field(ge=1, le=9)


class StrictFlightSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str
    destination: str


class BookFlightSchema(BaseModel):
    flight_id: str
    payment_method: Literal["credit_card", "paypal"]


def _test_case(tools_called):
    return LLMTestCase(
        input="Book a flight",
        actual_output="Booking confirmed",
        tools_called=tools_called,
    )


class TestToolArgumentValidationMetric:
    """ToolArgumentValidationMetric is deterministic, requiring zero LLM tokens and no API keys."""

    def test_all_calls_conform_passes(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={
                "search_flights": SearchFlightSchema,
                "book_flight": BookFlightSchema,
            }
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 2,
                    },
                ),
                ToolCall(
                    name="book_flight",
                    input_parameters={
                        "flight_id": "AZ123",
                        "payment_method": "credit_card",
                    },
                ),
            ]
        )
        score = metric.measure(test_case)
        assert score == 1.0
        assert metric.is_successful() is True
        assert "All 2 tool call(s) conformed" in metric.reason
        assert len(metric.validation_results) == 2
        assert all(r.is_valid for r in metric.validation_results)

    def test_type_mismatch_fails(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": "two_passengers",  # invalid int
                    },
                )
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "passengers" in metric.reason
        assert len(metric.validation_results[0].errors) > 0

    def test_missing_required_field_fails(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN"
                    },  # missing destination & passengers
                )
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "destination" in metric.reason

    def test_field_constraint_violation(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 15,  # exceeds le=9 constraint
                    },
                )
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "passengers" in metric.reason

    def test_forbidden_extra_arguments_fails(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_strict": StrictFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_strict",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "hallucinated_param": "unexpected",
                    },
                )
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "hallucinated_param" in metric.reason

    def test_json_string_input_parameters(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"book_flight": BookFlightSchema}
        )
        # Valid JSON string (e.g. constructed via model_construct or external tracers)
        valid_case = _test_case(
            [
                ToolCall.model_construct(
                    name="book_flight",
                    input_parameters='{"flight_id": "FL456", "payment_method": "paypal"}',
                )
            ]
        )
        assert metric.measure(valid_case) == 1.0

        # Malformed JSON string
        malformed_case = _test_case(
            [
                ToolCall.model_construct(
                    name="book_flight",
                    input_parameters="{not_valid_json: 123}",
                )
            ]
        )
        assert metric.measure(malformed_case) == 0.0
        assert "json_invalid" in metric.reason

    def test_no_tools_called_passes(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case([])
        score = metric.measure(test_case)
        assert score == 1.0
        assert metric.is_successful() is True
        assert metric.reason == "No tool calls provided"

    def test_unregistered_tool_fails_by_default(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [ToolCall(name="unregistered_tool", input_parameters={})]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False
        assert "not registered" in metric.reason

    def test_unregistered_tool_ignored_when_allowed(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema},
            allow_unregistered=True,
        )
        # 1 valid registered call, 1 unregistered call ignored -> 1/1 = 1.0
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 1,
                    },
                ),
                ToolCall(name="unregistered_tool", input_parameters={}),
            ]
        )
        score = metric.measure(test_case)
        assert score == 1.0
        assert metric.is_successful() is True

    def test_partial_score_with_threshold(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={
                "search_flights": SearchFlightSchema,
                "book_flight": BookFlightSchema,
            },
            threshold=0.5,
        )
        # 1 valid, 1 invalid -> 0.5; passes at threshold 0.5
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 2,
                    },
                ),
                ToolCall(
                    name="book_flight",
                    input_parameters={
                        "flight_id": "AZ123"
                    },  # missing payment_method
                ),
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.5
        assert metric.is_successful() is True

    def test_strict_mode_zeroes_partial_score(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={
                "search_flights": SearchFlightSchema,
                "book_flight": BookFlightSchema,
            },
            strict_mode=True,
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 2,
                    },
                ),
                ToolCall(
                    name="book_flight",
                    input_parameters={
                        "flight_id": "AZ123"
                    },  # missing payment_method
                ),
            ]
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert metric.is_successful() is False

    def test_invalid_tool_schemas_initialization(self):
        with pytest.raises(ValueError):
            ToolArgumentValidationMetric(tool_schemas={})

        with pytest.raises(TypeError):
            ToolArgumentValidationMetric(
                tool_schemas={
                    "bad_tool": dict
                }  # dict is not BaseModel subclass
            )

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 2,
                    },
                )
            ]
        )
        score = await metric.a_measure(test_case)
        assert score == 1.0
        assert metric.is_successful() is True

    def test_verbose_mode_logging(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema},
            verbose_mode=True,
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": 2,
                    },
                )
            ]
        )
        metric.measure(test_case)
        assert metric.verbose_logs is not None
        assert "search_flights" in metric.verbose_logs

    def test_nested_pydantic_model_validation(self):
        """Nested schemas are common in real tool signatures (e.g. address inside a booking)."""

        class Address(BaseModel):
            street: str
            city: str

        class CreateBooking(BaseModel):
            guest_name: str
            address: Address
            tags: Optional[List[str]] = None

        metric = ToolArgumentValidationMetric(
            tool_schemas={"create_booking": CreateBooking}
        )

        # Valid nested input
        valid_case = _test_case(
            [
                ToolCall(
                    name="create_booking",
                    input_parameters={
                        "guest_name": "Alice",
                        "address": {"street": "Via Roma 1", "city": "Turin"},
                    },
                )
            ]
        )
        assert metric.measure(valid_case) == 1.0

        # Invalid nested input: missing required nested field 'city'
        invalid_case = _test_case(
            [
                ToolCall(
                    name="create_booking",
                    input_parameters={
                        "guest_name": "Alice",
                        "address": {"street": "Via Roma 1"},
                    },
                )
            ]
        )
        assert metric.measure(invalid_case) == 0.0
        assert metric.validation_results[0].errors[0].field_path == "address.city"

    def test_tools_called_none_raises_missing_params(self):
        """When tools_called is None the framework must reject the test case."""
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = LLMTestCase(
            input="Book a flight", actual_output="Booking confirmed"
        )  # tools_called defaults to None
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(test_case)

    def test_include_reason_false_suppresses_reason(self):
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema},
            include_reason=False,
        )
        test_case = _test_case(
            [
                ToolCall(
                    name="search_flights",
                    input_parameters={
                        "origin": "TRN",
                        "destination": "LON",
                        "passengers": "bad",
                    },
                )
            ]
        )
        metric.measure(test_case)
        assert metric.score == 0.0
        assert metric.reason is None

    def test_none_input_parameters_treated_as_empty_dict(self):
        """External tracers may produce ToolCalls with input_parameters=None."""
        metric = ToolArgumentValidationMetric(
            tool_schemas={"search_flights": SearchFlightSchema}
        )
        test_case = _test_case(
            [
                ToolCall.model_construct(
                    name="search_flights", input_parameters=None
                )
            ]
        )
        score = metric.measure(test_case)
        # All required fields (origin, destination, passengers) are missing
        assert score == 0.0
        assert len(metric.validation_results[0].errors) >= 2
