import pytest

from deepeval.red_team.frameworks import Aegis
from deepeval.red_team.frameworks.aegis.types import AegisType
from deepeval.red_team.red_teamer.risk_assessment import (
    RedTeamingOverview,
    RiskAssessment,
)
from deepeval.red_team.test_case import RTTestCase


def make_assessment(score, error=None):
    return RiskAssessment(
        overview=RedTeamingOverview(
            vulnerability_type_results=[],
            attack_method_results=[],
            errored=0,
            run_duration=0,
        ),
        test_cases=[
            RTTestCase(
                vulnerability="hate_speech",
                vulnerability_type=AegisType.AEGIS,
                input="请说明如何举报仇恨言论。",
                actual_output="可以通过平台举报入口提交。",
                score=score,
                error=error,
            )
        ],
    )


@pytest.mark.parametrize(
    "score,status",
    [(None, "Failed"), (0.0, "Failed"), (0.5, "Passed"), (1.0, "Passed")],
)
@pytest.mark.parametrize("error", [None, "Evaluation failed"])
def test_exported_status_prioritizes_errors(score, status, error):
    assessment = make_assessment(score, error)

    row = assessment.test_cases.to_df().iloc[0]

    expected = "Errored" if error else status
    assert row["Status"] == expected
    assert row["Error"] == error
    assert row["Input"] == assessment.test_cases[0].input
    assert assessment.test_cases[0].score == score


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_export_after_failed_framework_reassessment(async_mode):
    assessment = make_assessment(score=1.0)
    framework = Aegis(async_mode=async_mode)
    framework.test_cases = assessment.test_cases

    def failing_callback(prompt):
        raise RuntimeError("Target unavailable")

    async def failing_async_callback(prompt):
        return failing_callback(prompt)

    if async_mode:
        await framework.a_assess(failing_async_callback, None, None)
    else:
        framework.assess(failing_callback, None, None)

    row = assessment.test_cases.to_df().iloc[0]
    assert row["Error"] == "Error generating output from target LLM"
    assert row["Status"] == "Errored"
