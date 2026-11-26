from deepeval.metrics.arena_g_eval.utils import format_arena_test_case
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)

KOREAN_INPUT = "여권 재발급하려면 어떤 서류가 필요한가요?"
KOREAN_OUTPUT = "신분증과 사진이 필요합니다."


def test_arena_prompt_keeps_non_ascii_text_readable():
    test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="a",
                test_case=LLMTestCase(
                    input=KOREAN_INPUT, actual_output=KOREAN_OUTPUT
                ),
            ),
            Contestant(
                name="b",
                test_case=LLMTestCase(
                    input=KOREAN_INPUT, actual_output="신청서만 있으면 됩니다."
                ),
            ),
        ]
    )
    formatted, _ = format_arena_test_case(
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], test_case
    )

    prompt_section = repr(formatted)

    assert KOREAN_INPUT in prompt_section
    assert KOREAN_OUTPUT in prompt_section
    assert "\\u" not in prompt_section
