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


def create_logprob_token(token, logprob, alternatives):
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[
            SimpleNamespace(token=t, logprob=p) for t, p in alternatives
        ],
    )


def test_calculate_weighted_summed_score_drops_digit_split_ambiguous_prefix():
    # Raw 9; the "1" alternative is the first token of a possible 10 under a
    # digit-splitting tokenizer, not the score 1. It must not drag the
    # weighted score toward 1 (#3517): 7.69 before, ~8.97 after.
    import math

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
    # Raw 10, emitted as "1","0". P(10) = P("1") * P("0" | "1") = 0.7 * 0.95;
    # the single-token alternative "9" carries 0.3. Expected ~9.69 (#3517);
    # this case crashed with AttributeError before.
    import math

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
    # The response never spells the score (raw 10, but no "10" and no
    # "1","0" pair anywhere): nothing to weight, so the raw score stands.
    raw_response = create_raw_response([create_token("7", "7")])

    assert calculate_weighted_summed_score(10, raw_response) == 10


def test_calculate_weighted_summed_score_keeps_prefix_when_range_is_single_digit():
    # With an explicit single-digit maximum, "1" can only be the score 1, so
    # the full distribution weights again (#3517's rescaling workaround,
    # exact instead of approximate).
    import math

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
    import math

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
