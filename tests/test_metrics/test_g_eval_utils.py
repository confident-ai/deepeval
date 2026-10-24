from types import SimpleNamespace

import pytest

from deepeval.errors import MissingTestCaseParamsError
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


def create_token_with_alts(token, alternatives):
    """top_logprobs with several candidates. `alternatives` maps a token
    string to its linear probability; logprob is stored as ln(probability)."""
    import math

    return SimpleNamespace(
        token=token,
        top_logprobs=[
            SimpleNamespace(token=t, logprob=math.log(p))
            for t, p in alternatives.items()
        ],
    )


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


def test_calculate_weighted_summed_score_ignores_multi_digit_prefix():
    # The judge hovers between 9 and 8 but also considers 10. A tokenizer that
    # splits 10 into "1"+"0" puts "1" (a prefix of 10) at the score position;
    # it must not be weighted as a complete score of 1.
    raw_response = create_raw_response(
        [create_token_with_alts("9", {"9": 0.812, "1": 0.16, "8": 0.028})]
    )

    # The "1" candidate (16% mass) is dropped and the rest are renormalized
    # to 1, so the weighted score ignores the ambiguous multi-digit prefix.
    assert calculate_weighted_summed_score(9, raw_response) == pytest.approx(
        (9 * 0.812 + 8 * 0.028) / (0.812 + 0.028)
    )


def test_calculate_weighted_summed_score_falls_back_for_multi_token_score():
    # A raw score of 10 is emitted as "1" then "0", so no single token equals
    # "10". There are no usable score logprobs: fall back to the raw score
    # instead of failing.
    raw_response = create_raw_response(
        [
            create_token_with_alts("1", {"1": 0.7, "9": 0.3}),
            create_token("0", "0"),
        ]
    )

    assert calculate_weighted_summed_score(10, raw_response) == 10


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
