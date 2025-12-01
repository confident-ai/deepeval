import pytest

from tests.test_core.helpers import _extract_example_json
from deepeval.metrics.g_eval.template import GEvalTemplate


def test_g_eval_generate_evaluation_steps_example_json_is_valid():
    """
    Example JSON in GEvalTemplate.generate_evaluation_steps must parse.

    Previously this example used a placeholder like <list_of_strings>,
    which is not valid JSON and caused json.loads to fail. This test
    guarantees the example is real, parseable JSON with the expected shape.
    """
    template = GEvalTemplate.generate_evaluation_steps(
        parameters="answer and reference",
        criteria="check factual correctness",
    )
    data = _extract_example_json(template)
    assert isinstance(data, dict)
    assert "steps" in data
    assert isinstance(data["steps"], list)


@pytest.mark.parametrize(
    "template",
    [
        GEvalTemplate.generate_evaluation_steps(
            parameters="answer and reference",
            criteria="check factual correctness",
        ),
        GEvalTemplate.generate_evaluation_results(
            evaluation_steps="1) check; 2) score",
            test_case_content="model output here",
            parameters="answer and reference",
        ),
        GEvalTemplate.generate_strict_evaluation_results(
            evaluation_steps="1) check; 2) score",
            test_case_content="model output here",
            parameters="answer and reference",
        ),
    ],
)
def test_g_eval_templates_emphasize_valid_json_and_only_json(template: str):
    """
    GEval templates should clearly ask for valid, parseable JSON,
    and for returning ONLY the JSON.
    """
    lower = template.lower()
    # New, stronger instruction you added
    assert "valid and parseable json" in lower
    # All three templates include "only return ..." in their IMPORTANT blocks
    assert "only return" in lower
