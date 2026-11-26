import json

from deepeval.metrics.arena_g_eval.utils import format_arena_test_case
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
)

KOREAN_INPUT = "서울의 날씨는 어때요?"
KOREAN_OUTPUT = '오늘 서울은 "맑음"입니다.\n기온은 20도예요.'
OTHER_OUTPUT = "비가 와요."

PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
]


def _arena_test_case():
    return ArenaTestCase(
        contestants=[
            Contestant(
                name="Version 1",
                test_case=LLMTestCase(
                    input=KOREAN_INPUT, actual_output=KOREAN_OUTPUT
                ),
            ),
            Contestant(
                name="Version 2",
                test_case=LLMTestCase(
                    input=KOREAN_INPUT, actual_output=OTHER_OUTPUT
                ),
            ),
        ]
    )


def test_judge_prompt_keeps_non_ascii_text_readable():
    formatted, _ = format_arena_test_case(PARAMS, _arena_test_case())

    prompt = repr(formatted)

    assert KOREAN_INPUT in prompt
    assert "서울은" in prompt
    assert "\\u" not in prompt


def test_contestants_are_nested_as_objects_not_json_strings():
    formatted, dummy_to_real_names = format_arena_test_case(
        PARAMS, _arena_test_case()
    )

    data = json.loads(repr(formatted))

    assert data["input"] == KOREAN_INPUT
    outputs = {}
    for dummy_name, contestant in data["arena_test_cases"].items():
        assert isinstance(contestant, dict)
        outputs[dummy_to_real_names[dummy_name]] = contestant["actual_output"]
    assert outputs == {"Version 1": KOREAN_OUTPUT, "Version 2": OTHER_OUTPUT}


def test_quotes_and_newlines_are_escaped_once():
    formatted, _ = format_arena_test_case(PARAMS, _arena_test_case())

    prompt = repr(formatted)

    assert '\\"맑음\\"' in prompt
    assert '\\\\"' not in prompt
    assert "\\\\n" not in prompt


def test_single_test_case_repr_is_readable_json():
    formatted, _ = format_arena_test_case(PARAMS, _arena_test_case())
    contestant = next(iter(formatted.contestants.values()))

    text = repr(contestant)

    assert "\\u" not in text
    assert json.loads(text)["actual_output"] in (KOREAN_OUTPUT, OTHER_OUTPUT)


def test_tool_calls_are_nested_as_objects_and_escaped_once():
    tool = ToolCall(
        name="검색",
        input_parameters={"q": '서울 "날씨"\n오늘'},
        output='결과\n"맑음"',
    )
    test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="Version 1",
                test_case=LLMTestCase(
                    input=KOREAN_INPUT,
                    actual_output=OTHER_OUTPUT,
                    tools_called=[tool],
                    expected_tools=[tool],
                ),
            ),
        ]
    )
    formatted, _ = format_arena_test_case(
        [SingleTurnParams.TOOLS_CALLED, SingleTurnParams.EXPECTED_TOOLS],
        test_case,
    )

    prompt = repr(formatted)
    contestant = next(iter(json.loads(prompt)["arena_test_cases"].values()))

    for field in ("tools_called", "expected_tools"):
        assert contestant[field] == [
            {
                "name": "검색",
                "type": "FUNCTION",
                "input_parameters": {"q": '서울 "날씨"\n오늘'},
                "output": '결과\n"맑음"',
            }
        ]
    assert '\\\\"' not in prompt
    assert "\\\\n" not in prompt
    assert "\\u" not in prompt
