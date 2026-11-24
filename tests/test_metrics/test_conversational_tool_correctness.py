import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import ConversationalToolCorrectnessMetric
from deepeval.test_case import (
    ConversationalTestCase,
    ToolCall,
    ToolCallParams,
    Turn,
)


def _track(number):
    return ToolCall(name="track_parcel", input_parameters={"pin": number})


def _transfer():
    return ToolCall(name="transfer_to_human", input_parameters={})


def _conversation(*assistant_turns):
    turns = []
    for tools_called, expected_tools in assistant_turns:
        turns.append(Turn(role="user", content="hi"))
        turns.append(
            Turn(
                role="assistant",
                content="ok",
                tools_called=tools_called,
                expected_tools=expected_tools,
            )
        )
    return ConversationalTestCase(turns=turns)


def test_averages_only_turns_with_expected_tools():
    metric = ConversationalToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            ([_track("123")], [_track("123")]),
            ([_track("123")], [_transfer()]),
            ([_transfer()], None),  # not scored
        ),
        _show_indicator=False,
    )
    assert score == 0.5
    assert [t["turn"] for t in metric.turn_scores] == [1, 3]
    assert metric.success is True
    assert "1 of 2" in metric.reason and "Turn 3" in metric.reason


def test_empty_expected_tools_means_no_calls_expected():
    metric = ConversationalToolCorrectnessMetric()
    assert metric.measure(_conversation(([], [])), _show_indicator=False) == 1
    assert (
        metric.measure(
            _conversation(([_transfer()], [])), _show_indicator=False
        )
        == 0
    )


def test_input_parameters_are_compared_when_asked():
    case = _conversation(([_track("999")], [_track("123")]))
    assert (
        ConversationalToolCorrectnessMetric().measure(
            case, _show_indicator=False
        )
        == 1
    )
    assert (
        ConversationalToolCorrectnessMetric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ).measure(case, _show_indicator=False)
        == 0
    )


def test_strict_mode_zeroes_a_partial_score():
    metric = ConversationalToolCorrectnessMetric(strict_mode=True)
    case = _conversation(([_track("1")], [_track("1")]), ([], [_transfer()]))
    assert metric.measure(case, _show_indicator=False) == 0
    assert metric.success is False


def test_errors_without_any_expected_tools():
    metric = ConversationalToolCorrectnessMetric()
    with pytest.raises(MissingTestCaseParamsError, match="expected_tools"):
        metric.measure(_conversation(([_track("1")], None)))


def test_expected_tools_round_trips_through_aliases_and_api():
    from deepeval.test_case.api import create_api_turn

    turn = Turn.model_validate(
        {
            "role": "assistant",
            "content": "ok",
            "expectedTools": [{"name": "track_parcel"}],
        }
    )
    assert turn.expected_tools[0].name == "track_parcel"
    api_turn = create_api_turn(turn, 0).model_dump(by_alias=True)
    assert api_turn["expectedTools"][0]["name"] == "track_parcel"


async def test_a_measure_matches_measure():
    case = _conversation(([_track("1")], [_track("1")]))
    assert (
        await ConversationalToolCorrectnessMetric().a_measure(
            case, _show_indicator=False
        )
        == 1
    )
