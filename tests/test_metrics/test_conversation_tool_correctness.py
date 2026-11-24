import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.dataset.utils import convert_convo_goldens_to_convo_test_cases
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import ConversationToolCorrectnessMetric
from deepeval.test_case import (
    ConversationalTestCase,
    ToolCall,
    ToolCallParams,
    Turn,
)


def _search(destination):
    return ToolCall(
        name="search_flights", input_parameters={"destination": destination}
    )


def _book():
    return ToolCall(name="book_flight", input_parameters={})


def _conversation(*tools_per_assistant_turn, expected_tools=None):
    turns = []
    for tools_called in tools_per_assistant_turn:
        turns.append(Turn(role="user", content="hi"))
        turns.append(
            Turn(role="assistant", content="ok", tools_called=tools_called)
        )
    return ConversationalTestCase(turns=turns, expected_tools=expected_tools)


def test_collects_tools_across_all_assistant_turns():
    metric = ConversationToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            [_search("123")],
            None,
            [_book()],
            expected_tools=[_search("123"), _book()],
        ),
        _show_indicator=False,
    )
    assert score == 1
    assert [t.name for t in metric.tools_called] == [
        "search_flights",
        "book_flight",
    ]
    assert metric.success is True


def test_missing_expected_tool_lowers_the_score():
    metric = ConversationToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            [_search("123")], [], expected_tools=[_search("123"), _book()]
        ),
        _show_indicator=False,
    )
    assert score == 0.5
    assert "book_flight" in metric.reason


def test_user_turn_tools_are_ignored():
    test_case = _conversation([], expected_tools=[_book()])
    test_case.turns[0] = Turn(role="user", content="hi", tools_called=[_book()])
    metric = ConversationToolCorrectnessMetric()
    assert metric.measure(test_case, _show_indicator=False) == 0


def test_empty_expected_tools_means_no_calls_expected():
    metric = ConversationToolCorrectnessMetric()
    assert (
        metric.measure(
            _conversation([], [], expected_tools=[]), _show_indicator=False
        )
        == 1
    )
    assert (
        metric.measure(
            _conversation([_book()], expected_tools=[]),
            _show_indicator=False,
        )
        == 0
    )


def test_ordering_is_across_the_conversation():
    expected = [_search("123"), _book()]
    in_order = _conversation(
        [_search("123")], [_book()], expected_tools=expected
    )
    reversed_order = _conversation(
        [_book()], [_search("123")], expected_tools=expected
    )
    metric = ConversationToolCorrectnessMetric(should_consider_ordering=True)
    assert metric.measure(in_order, _show_indicator=False) == 1
    assert metric.measure(reversed_order, _show_indicator=False) < 1


def test_exact_match_rejects_extra_calls():
    metric = ConversationToolCorrectnessMetric(should_exact_match=True)
    score = metric.measure(
        _conversation(
            [_search("123")], [_search("123")], expected_tools=[_search("123")]
        ),
        _show_indicator=False,
    )
    assert score == 0


def test_input_parameters_are_compared_when_asked():
    case = _conversation([_search("999")], expected_tools=[_search("123")])
    assert (
        ConversationToolCorrectnessMetric().measure(case, _show_indicator=False)
        == 1
    )
    assert (
        ConversationToolCorrectnessMetric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ).measure(case, _show_indicator=False)
        == 0
    )


def test_strict_mode_zeroes_a_partial_score():
    metric = ConversationToolCorrectnessMetric(strict_mode=True)
    score = metric.measure(
        _conversation(
            [_search("123")], expected_tools=[_search("123"), _book()]
        ),
        _show_indicator=False,
    )
    assert score == 0
    assert metric.success is False


def test_errors_without_expected_tools():
    metric = ConversationToolCorrectnessMetric()
    with pytest.raises(MissingTestCaseParamsError, match="expected_tools"):
        metric.measure(_conversation([_search("123")]), _show_indicator=False)


def test_golden_expected_tools_reach_the_test_case():
    golden = ConversationalGolden(
        scenario="User wants to book a flight.",
        turns=[Turn(role="user", content="Book me a flight to Tokyo.")],
        expected_tools=[{"name": "book_flight"}],
    )
    assert (
        golden.model_dump(by_alias=True)["expectedTools"][0]["name"]
        == "book_flight"
    )
    (test_case,) = convert_convo_goldens_to_convo_test_cases([golden])
    assert [t.name for t in test_case.expected_tools] == ["book_flight"]


def test_expected_tools_round_trips_through_aliases_and_api():
    from deepeval.test_case.api import create_api_test_case

    test_case = ConversationalTestCase.model_validate(
        {
            "turns": [{"role": "user", "content": "hi"}],
            "expectedTools": [{"name": "search_flights"}],
        }
    )
    assert test_case.expected_tools[0].name == "search_flights"
    api = create_api_test_case(test_case, index=0).model_dump(by_alias=True)
    assert api["expectedTools"][0]["name"] == "search_flights"


async def test_a_measure_matches_measure():
    case = _conversation([_search("1")], expected_tools=[_search("1")])
    assert (
        await ConversationToolCorrectnessMetric().a_measure(
            case, _show_indicator=False
        )
        == 1
    )


def _pinned(tool, turn):
    return tool.model_copy(update={"turn": turn})


def test_pinned_tool_must_be_called_on_its_turn():
    on_time = _conversation(
        [_search("1")], [_book()], expected_tools=[_pinned(_book(), 3)]
    )
    too_early = _conversation(
        [_book()], [], expected_tools=[_pinned(_book(), 3)]
    )
    metric = ConversationToolCorrectnessMetric()
    assert metric.measure(on_time, _show_indicator=False) == 1
    assert metric.measure(too_early, _show_indicator=False) == 0
    assert "book_flight" in metric.reason


def test_pinned_and_unpinned_tools_are_scored_separately():
    metric = ConversationToolCorrectnessMetric()
    score = metric.measure(
        _conversation(
            [_search("1")],
            [],
            expected_tools=[_search("1"), _pinned(_book(), 3)],
        ),
        _show_indicator=False,
    )
    assert score == 0.5
    assert [c["check"] for c in metric.check_scores] == [
        "Turn 3",
        "Rest of the conversation",
    ]
    assert "Turn 3 (0.00)" in metric.reason


def test_pinned_turn_is_left_out_of_the_rest_of_the_conversation():
    metric = ConversationToolCorrectnessMetric(should_exact_match=True)
    score = metric.measure(
        _conversation(
            [_search("1")],
            [_book()],
            expected_tools=[_search("1"), _pinned(_book(), 3)],
        ),
        _show_indicator=False,
    )
    assert score == 1


def test_turn_must_point_at_an_assistant_turn():
    metric = ConversationToolCorrectnessMetric()
    with pytest.raises(
        MissingTestCaseParamsError, match="not an assistant turn"
    ):
        metric.measure(
            _conversation([_book()], expected_tools=[_pinned(_book(), 0)]),
            _show_indicator=False,
        )
    with pytest.raises(MissingTestCaseParamsError, match="only has 2 turns"):
        metric.measure(
            _conversation([_book()], expected_tools=[_pinned(_book(), 5)]),
            _show_indicator=False,
        )


def test_turn_does_not_change_tool_equality_or_unset_payloads():
    assert _pinned(_book(), 3) == _book()
    assert "turn" not in _book().model_dump(by_alias=True, exclude_none=True)
    assert (
        _pinned(_book(), 3).model_dump(by_alias=True, exclude_none=True)["turn"]
        == 3
    )
