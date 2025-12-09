# tests/test_core/test_optimization/test_mutations/test_prompt_rewriter.py
import pytest

from deepeval.optimizer.rewriter.utils import (
    _compose_prompt_messages,
    _normalize_llm_output_to_text,
)

###############################
# _compose_prompt_messages    #
###############################


@pytest.mark.parametrize(
    "system,user,expected",
    [
        ("System", "User", "System\n\nUser"),
        ("", "User only", "User only"),
        ("   ", "  Trim me  ", "Trim me"),
    ],
)
def test_compose_prompt_messages(system, user, expected):
    assert _compose_prompt_messages(system, user) == expected


########################################
# _normalize_llm_output_to_text        #
########################################


def test_normalize_llm_output_to_text_str_and_tuple():
    assert _normalize_llm_output_to_text("  hi  ") == "hi"

    text = _normalize_llm_output_to_text(("  hello ", 1.23))
    assert text == "hello"


def test_normalize_llm_output_to_text_dict_json_and_unicode():
    data = {"answer": "café", "score": 0.9}
    out = _normalize_llm_output_to_text(data)

    # Should be JSON and preserve unicode characters
    assert '"answer"' in out
    assert "café" in out
    assert '"score"' in out


def test_normalize_llm_output_to_text_unserializable_falls_back_to_str():
    class Unserializable:
        def __repr__(self):
            return "<UnserializableObject>"

    obj = Unserializable()
    out = _normalize_llm_output_to_text(obj)

    assert isinstance(out, str)
    assert "UnserializableObject" in out
