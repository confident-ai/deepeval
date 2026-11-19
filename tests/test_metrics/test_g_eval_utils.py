import math
from types import SimpleNamespace

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics.g_eval.utils import (
    CONVERSATIONAL_G_EVAL_API_PARAMS,
    G_EVAL_API_PARAMS,
    construct_geval_pull_evaluation_params,
    construct_geval_upload_payload,
    construct_non_turns_test_case_string,
    construct_test_case_string,
    calculate_weighted_summed_score,
)
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    check_llm_test_case_params,
    convert_turn_to_dict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    SingleTurnParams,
    Turn,
    MultiTurnParams,
)


class DummyMetric:
    __name__ = "DummyMetric"
    error = None


class DummyConversationalMetric:
    __name__ = "DummyConversationalMetric"
    error = None


def create_raw_response(tokens):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(content=tokens),
            )
        ]
    )


def create_token(token, top_token):
    return SimpleNamespace(
        token=token,
        top_logprobs=[SimpleNamespace(token=top_token, logprob=0)],
    )


def create_logprob_token(token, logprob, alternatives):
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[
            SimpleNamespace(token=t, logprob=p) for t, p in alternatives
        ],
    )


def test_calculate_weighted_summed_score_drops_digit_split_ambiguous_prefix():
    # "1" may be the first digit of 10, so it must not count as a score of 1.
    raw_response = create_raw_response(
        [
            create_logprob_token("score", 0, [("score", 0)]),
            create_logprob_token(
                "9",
                math.log(0.812),
                [
                    ("9", math.log(0.812)),
                    ("1", math.log(0.16)),
                    ("8", math.log(0.028)),
                ],
            ),
            create_logprob_token("}", 0, [("}", 0)]),
        ]
    )

    weighted = calculate_weighted_summed_score(9, raw_response)
    assert weighted == pytest.approx(
        (9 * 0.812 + 8 * 0.028) / (0.812 + 0.028), abs=0.01
    )


def test_calculate_weighted_summed_score_weights_multitoken_raw_score():
    # 10 emitted as "1", "0": P(10) = P("1") * P("0" | "1").
    raw_response = create_raw_response(
        [
            create_logprob_token("score", 0, [("score", 0)]),
            create_logprob_token(
                "1", math.log(0.7), [("1", math.log(0.7)), ("9", math.log(0.3))]
            ),
            create_logprob_token("0", math.log(0.95), [("0", math.log(0.95))]),
            create_logprob_token("}", 0, [("}", 0)]),
        ]
    )

    weighted = calculate_weighted_summed_score(10, raw_response)
    assert weighted == pytest.approx(
        (10 * 0.7 * 0.95 + 9 * 0.3) / (0.7 * 0.95 + 0.3), abs=0.01
    )


def test_calculate_weighted_summed_score_returns_raw_when_span_missing():
    raw_response = create_raw_response([create_token("7", "7")])

    assert calculate_weighted_summed_score(10, raw_response) == 10


def test_calculate_weighted_summed_score_keeps_prefix_when_range_is_single_digit():
    raw_response = create_raw_response(
        [
            create_logprob_token(
                "2", math.log(0.6), [("2", math.log(0.6)), ("1", math.log(0.4))]
            ),
        ]
    )

    weighted = calculate_weighted_summed_score(2, raw_response, score_max=9)
    assert weighted == pytest.approx((2 * 0.6 + 1 * 0.4) / 1.0, abs=0.01)


def test_calculate_weighted_summed_score_ignores_out_of_range_candidates():
    raw_response = create_raw_response(
        [
            create_logprob_token(
                "9",
                math.log(0.5),
                [("9", math.log(0.5)), ("11", math.log(0.5))],
            ),
        ]
    )

    assert calculate_weighted_summed_score(9, raw_response) == 9


def test_calculate_weighted_summed_score_ignores_candidates_below_score_min():
    # A 1-5 rubric: the judge puts 30% on "0", which is below the minimum.
    raw_response = create_raw_response(
        [
            create_logprob_token(
                "1",
                math.log(0.7),
                [("1", math.log(0.7)), ("0", math.log(0.3))],
            ),
        ]
    )

    weighted = calculate_weighted_summed_score(
        1, raw_response, score_max=5, score_min=1
    )
    assert weighted == pytest.approx(1.0, abs=0.01)


def test_calculate_weighted_summed_score_counts_zero_by_default():
    # Without a rubric minimum, "0" is still a valid candidate.
    raw_response = create_raw_response(
        [
            create_logprob_token(
                "1",
                math.log(0.7),
                [("1", math.log(0.7)), ("0", math.log(0.3))],
            ),
        ]
    )

    weighted = calculate_weighted_summed_score(1, raw_response)
    assert weighted == pytest.approx(0.7, abs=0.01)


class MockJudge(DeepEvalBaseLLM):
    """Returns score 1 with 70% probability and puts 30% on 0."""

    def __init__(self):
        super().__init__("mock-judge")

    def load_model(self):
        return None

    def get_model_name(self):
        return "mock-judge"

    def generate(self, prompt, *a, **k):
        return '{"reason": "weak", "score": 1}'

    async def a_generate(self, prompt, *a, **k):
        return self.generate(prompt, *a, **k)

    def _raw(self):
        token = SimpleNamespace(
            token="1",
            logprob=math.log(0.7),
            top_logprobs=[
                SimpleNamespace(token="1", logprob=math.log(0.7)),
                SimpleNamespace(token="0", logprob=math.log(0.3)),
            ],
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"reason": "weak", "score": 1}'
                    ),
                    logprobs=SimpleNamespace(content=[token]),
                )
            ]
        )

    def generate_raw_response(self, prompt, top_logprobs=None, **k):
        return self._raw(), 0.0

    async def a_generate_raw_response(self, prompt, top_logprobs=None, **k):
        return self._raw(), 0.0


def test_geval_score_stays_in_range_when_rubric_starts_above_zero():
    metric = GEval(
        name="Helpfulness",
        evaluation_steps=["Judge how helpful the answer is."],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        rubric=[
            Rubric(score_range=(1, 2), expected_outcome="Unhelpful."),
            Rubric(score_range=(3, 5), expected_outcome="Helpful."),
        ],
        model=MockJudge(),
        async_mode=False,
    )

    score = metric.measure(
        LLMTestCase(input="q", actual_output="a"), _show_indicator=False
    )
    assert score == pytest.approx(0.0, abs=0.01)


def test_calculate_weighted_summed_score_uses_final_score_token():
    raw_response = create_raw_response(
        [
            create_token("9", "1"),
            create_token("because", "because"),
            create_token("9", "9"),
        ]
    )

    assert calculate_weighted_summed_score(9, raw_response) == 9


def test_calculate_weighted_summed_score_with_single_score_token():
    raw_response = create_raw_response([create_token("6", "4")])

    assert calculate_weighted_summed_score(6, raw_response) == 4


def test_geval_accepts_metadata_and_tags():
    test_case = LLMTestCase(
        input="input",
        metadata={"source": "unit"},
        tags=["tag"],
    )

    text = construct_test_case_string(
        [SingleTurnParams.METADATA, SingleTurnParams.TAGS],
        test_case,
    )
    payload = construct_geval_upload_payload(
        name="metadata-test",
        evaluation_params=[SingleTurnParams.METADATA, SingleTurnParams.TAGS],
        g_eval_api_params=G_EVAL_API_PARAMS,
        criteria="criteria",
    )

    assert "Metadata" in text
    assert "Tags" in text
    assert payload["evaluationParams"] == ["metadata", "tags"]


def test_geval_requires_metadata_when_selected():
    test_case = LLMTestCase(input="input", tags=["tag"])

    with pytest.raises(MissingTestCaseParamsError):
        check_llm_test_case_params(
            test_case,
            [SingleTurnParams.METADATA],
            None,
            None,
            DummyMetric(),
        )


def test_conversational_geval_accepts_metadata_and_tags():
    case_metadata = {"case": "metadata"}
    case_tags = ["tag"]
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        metadata=case_metadata,
        tags=case_tags,
    )

    non_turn_text = construct_non_turns_test_case_string(
        [MultiTurnParams.METADATA, MultiTurnParams.TAGS],
        test_case,
    )
    turn_dict = convert_turn_to_dict(
        test_case.turns[0],
        [
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
            MultiTurnParams.METADATA,
            MultiTurnParams.TAGS,
        ],
    )
    payload = construct_geval_upload_payload(
        name="conversational-metadata-test",
        evaluation_params=[MultiTurnParams.METADATA, MultiTurnParams.TAGS],
        g_eval_api_params=CONVERSATIONAL_G_EVAL_API_PARAMS,
        criteria="criteria",
        multi_turn=True,
    )

    assert "Metadata" in non_turn_text
    assert "case" in non_turn_text
    assert "Tags" in non_turn_text
    assert "tag" in non_turn_text
    assert "metadata" not in turn_dict
    assert "tags" not in turn_dict
    assert payload["evaluationParams"] == ["metadata", "tags"]


def test_conversational_geval_uploads_and_pulls_audio():
    evaluation_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.AUDIO,
    ]
    payload = construct_geval_upload_payload(
        name="conversational-audio-test",
        evaluation_params=evaluation_params,
        g_eval_api_params=CONVERSATIONAL_G_EVAL_API_PARAMS,
        criteria="criteria",
        multi_turn=True,
    )

    assert payload["evaluationParams"] == ["role", "content", "audio"]
    assert (
        construct_geval_pull_evaluation_params(
            payload["evaluationParams"], multi_turn=True
        )
        == evaluation_params
    )


def test_conversational_geval_requires_metadata_when_selected():
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        tags=["tag"],
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_conversational_test_case_params(
            test_case,
            [MultiTurnParams.METADATA],
            DummyConversationalMetric(),
        )


def test_conversational_geval_requires_tags_when_selected():
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        metadata={"case": "metadata"},
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_conversational_test_case_params(
            test_case,
            [MultiTurnParams.TAGS],
            DummyConversationalMetric(),
        )


def test_conversational_geval_accepts_context_user_description_and_chatbot_role():
    evaluation_params = [
        MultiTurnParams.CONTEXT,
        MultiTurnParams.USER_DESCRIPTION,
        MultiTurnParams.CHATBOT_ROLE,
    ]
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        context=["refund policy"],
        user_description="frustrated customer",
        chatbot_role="support agent",
    )

    check_conversational_test_case_params(
        test_case, evaluation_params, DummyConversationalMetric()
    )
    non_turn_text = construct_non_turns_test_case_string(
        evaluation_params, test_case
    )
    turn_dict = convert_turn_to_dict(test_case.turns[0], evaluation_params)
    payload = construct_geval_upload_payload(
        name="conversational-context-test",
        evaluation_params=evaluation_params,
        g_eval_api_params=CONVERSATIONAL_G_EVAL_API_PARAMS,
        criteria="criteria",
        multi_turn=True,
    )

    assert "Context" in non_turn_text
    assert "refund policy" in non_turn_text
    assert "User Description" in non_turn_text
    assert "frustrated customer" in non_turn_text
    assert "Chatbot Role" in non_turn_text
    assert "support agent" in non_turn_text
    assert turn_dict == {}
    assert payload["evaluationParams"] == [
        "context",
        "userDescription",
        "chatbotRole",
    ]
    assert (
        construct_geval_pull_evaluation_params(
            payload["evaluationParams"], multi_turn=True
        )
        == evaluation_params
    )


@pytest.mark.parametrize(
    "param",
    [
        MultiTurnParams.CONTEXT,
        MultiTurnParams.USER_DESCRIPTION,
        MultiTurnParams.CHATBOT_ROLE,
    ],
)
def test_conversational_geval_requires_conversation_fields_when_selected(
    param,
):
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_conversational_test_case_params(
            test_case, [param], DummyConversationalMetric()
        )
