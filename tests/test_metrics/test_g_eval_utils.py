import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.g_eval.utils import (
    CONVERSATIONAL_G_EVAL_API_PARAMS,
    G_EVAL_API_PARAMS,
    construct_geval_upload_payload,
    construct_non_turns_test_case_string,
    construct_test_case_string,
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
