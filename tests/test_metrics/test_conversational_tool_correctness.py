import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.dataset.utils import convert_convo_goldens_to_convo_test_cases
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


def _conversation(*tools_per_assistant_turn, expected_tools=None):
    turns = []
    for tools_called in tools_per_assistant_turn:
        turns.append(Turn(role="user", content="hi"))
        turns.append(
            Turn(role="assistant", content="ok", tools_called=tools_called)
        )
    return ConversationalTestCase(turns=turns, expected_tools=expected_tools)


def test_collects_tools_across_all_assistant_turns():
    metric = ConversationalToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            [_track("123")],
            None,
            [_transfer()],
            expected_tools=[_track("123"), _transfer()],
        ),
        _show_indicator=False,
    )
    assert score == 1
    assert [t.name for t in metric.tools_called] == [
        "track_parcel",
        "transfer_to_human",
    ]
    assert metric.success is True


def test_missing_expected_tool_lowers_the_score():
    metric = ConversationalToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            [_track("123")], [], expected_tools=[_track("123"), _transfer()]
        ),
        _show_indicator=False,
    )
    assert score == 0.5
    assert "transfer_to_human" in metric.reason


def test_user_turn_tools_are_ignored():
    test_case = _conversation([], expected_tools=[_transfer()])
    test_case.turns[0] = Turn(
        role="user", content="hi", tools_called=[_transfer()]
    )
    metric = ConversationalToolCorrectnessMetric()
    assert metric.measure(test_case, _show_indicator=False) == 0


def test_empty_expected_tools_means_no_calls_expected():
    metric = ConversationalToolCorrectnessMetric()
    assert (
        metric.measure(
            _conversation([], [], expected_tools=[]), _show_indicator=False
        )
        == 1
    )
    assert (
        metric.measure(
            _conversation([_transfer()], expected_tools=[]),
            _show_indicator=False,
        )
        == 0
    )


def test_ordering_is_across_the_conversation():
    expected = [_track("123"), _transfer()]
    in_order = _conversation(
        [_track("123")], [_transfer()], expected_tools=expected
    )
    reversed_order = _conversation(
        [_transfer()], [_track("123")], expected_tools=expected
    )
    metric = ConversationalToolCorrectnessMetric(should_consider_ordering=True)
    assert metric.measure(in_order, _show_indicator=False) == 1
    assert metric.measure(reversed_order, _show_indicator=False) < 1


def test_exact_match_rejects_extra_calls():
    metric = ConversationalToolCorrectnessMetric(should_exact_match=True)
    score = metric.measure(
        _conversation(
            [_track("123")], [_track("123")], expected_tools=[_track("123")]
        ),
        _show_indicator=False,
    )
    assert score == 0


def test_input_parameters_are_compared_when_asked():
    case = _conversation([_track("999")], expected_tools=[_track("123")])
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
    score = metric.measure(
        _conversation(
            [_track("123")], expected_tools=[_track("123"), _transfer()]
        ),
        _show_indicator=False,
    )
    assert score == 0
    assert metric.success is False


def test_errors_without_expected_tools():
    metric = ConversationalToolCorrectnessMetric()
    with pytest.raises(MissingTestCaseParamsError, match="expected_tools"):
        metric.measure(_conversation([_track("123")]), _show_indicator=False)


def test_golden_expected_tools_reach_the_test_case():
    golden = ConversationalGolden(
        scenario="Parcel is late; user wants a human.",
        turns=[Turn(role="user", content="Get me a person.")],
        expected_tools=[{"name": "transfer_to_human"}],
    )
    assert (
        golden.model_dump(by_alias=True)["expectedTools"][0]["name"]
        == "transfer_to_human"
    )
    (test_case,) = convert_convo_goldens_to_convo_test_cases([golden])
    assert [t.name for t in test_case.expected_tools] == ["transfer_to_human"]


def test_expected_tools_round_trips_through_aliases_and_api():
    from deepeval.test_case.api import create_api_test_case

    test_case = ConversationalTestCase.model_validate(
        {
            "turns": [{"role": "user", "content": "hi"}],
            "expectedTools": [{"name": "track_parcel"}],
        }
    )
    assert test_case.expected_tools[0].name == "track_parcel"
    api = create_api_test_case(test_case, index=0).model_dump(by_alias=True)
    assert api["expectedTools"][0]["name"] == "track_parcel"


async def test_a_measure_matches_measure():
    case = _conversation([_track("1")], expected_tools=[_track("1")])
    assert (
        await ConversationalToolCorrectnessMetric().a_measure(
            case, _show_indicator=False
        )
        == 1
    )
