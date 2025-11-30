import pytest

from tests.test_core.helpers import _extract_example_json
from deepeval.metrics.conversational_g_eval.template import (
    ConversationalGEvalTemplate,
)


def test_conversational_g_eval_generate_evaluation_steps_example_json_is_valid():
    """Example JSON in ConversationalGEvalTemplate.generate_evaluation_steps must parse.

    On current main, the example used a placeholder like <list_of_strings>,
    which is not valid JSON and would cause json.loads(..) to fail. This test
    guards that the example remains real, valid JSON:

        {"steps": ["step 1", "step 2"]}
    """
    template = ConversationalGEvalTemplate.generate_evaluation_steps(
        parameters="role and content",
        criteria="check conversational quality and helpfulness",
    )
    data = _extract_example_json(template)
    assert isinstance(data, dict)
    assert "steps" in data
    assert isinstance(data["steps"], list)
    # Optional sanity check that it isn't empty
    assert len(data["steps"]) >= 1


def test_conversational_g_eval_generate_evaluation_results_example_json_is_valid():
    """Example JSON in ConversationalGEvalTemplate.generate_evaluation_results must parse.

    This ensures the documented example under "Example JSON:" is valid and
    matches the expected "score"/"reason" schema.
    """
    template = ConversationalGEvalTemplate.generate_evaluation_results(
        evaluation_steps="1) check; 2) score",
        test_case_content="conversation snippet here",
        turns=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        parameters="role and content",
        rubric=None,
    )
    data = _extract_example_json(template)
    assert isinstance(data, dict)
    assert "score" in data
    assert "reason" in data
    # Example JSON uses an integer score and a string reason
    assert isinstance(data["score"], int)
    assert isinstance(data["reason"], str)


@pytest.mark.parametrize(
    "template",
    [
        ConversationalGEvalTemplate.generate_evaluation_steps(
            parameters="role and content",
            criteria="check conversational quality and helpfulness",
        ),
        ConversationalGEvalTemplate.generate_evaluation_results(
            evaluation_steps="1) check; 2) score",
            test_case_content="conversation snippet here",
            turns=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            parameters="role and content",
            rubric=None,
        ),
    ],
)
def test_conversational_g_eval_templates_discourage_markdown_fences(
    template: str,
):
    """Conversational GEval templates should explicitly ask for valid JSON without fences."""
    lower = template.lower()
    assert "valid and parseable json" in lower
    assert "markdown code fences" in lower
