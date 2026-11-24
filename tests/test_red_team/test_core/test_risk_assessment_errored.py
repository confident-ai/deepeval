"""Regression test for ``RedTeamingOverview.errored``.

Before the fix, ``construct_risk_assessment_overview`` counted errored test
cases only to overwrite that counter inside the per-group loops with
``sum(1 for tc in test_cases if tc.error is not None)``. Those group lists
never contain errored cases (they are skipped before grouping), so the sum is
always 0 and ``overview.errored`` reported 0 for any run that had at least one
successful test case.

This test therefore fails on the pre-fix code with ``assert 0 == 2``.
"""

from deepeval.red_team.red_teamer.risk_assessment import (
    construct_risk_assessment_overview,
)
from deepeval.red_team.test_case import RTTestCase
from deepeval.red_team.vulnerabilities.bias.types import BiasType
from deepeval.red_team.vulnerabilities.toxicity.types import ToxicityType


def _test_case(vulnerability, vulnerability_type, score, error=None):
    return RTTestCase(
        vulnerability=vulnerability,
        vulnerability_type=vulnerability_type,
        input="attack prompt",
        actual_output=None if error else "target reply",
        score=score,
        error=error,
        attack_method="Base64",
    )


def test_overview_errored_matches_number_of_errored_test_cases():
    test_cases = [
        # Toxicity: 1 pass, 1 fail, 1 errored
        _test_case("Toxicity", ToxicityType.PROFANITY, 1),
        _test_case("Toxicity", ToxicityType.PROFANITY, 0),
        _test_case("Toxicity", ToxicityType.PROFANITY, None, "judge timeout"),
        # Bias: 1 pass, 1 errored
        _test_case("Bias", BiasType.GENDER, 1),
        _test_case("Bias", BiasType.GENDER, None, "attack simulation failed"),
    ]
    expected_errored = sum(1 for tc in test_cases if tc.error is not None)

    overview = construct_risk_assessment_overview(test_cases, run_duration=1.0)

    assert overview.errored == expected_errored == 2
    assert (
        sum(r.errored for r in overview.vulnerability_type_results)
        == expected_errored
    )
    assert (
        sum(r.errored for r in overview.attack_method_results)
        == expected_errored
    )


def test_errored_cases_do_not_change_pass_rate_denominator():
    """The fix must only restore the accounting, not move any score."""
    test_cases = [
        _test_case("Toxicity", ToxicityType.PROFANITY, 1),
        _test_case("Toxicity", ToxicityType.PROFANITY, 0),
        _test_case("Toxicity", ToxicityType.PROFANITY, None, "judge timeout"),
    ]

    overview = construct_risk_assessment_overview(test_cases, run_duration=1.0)
    result = overview.vulnerability_type_results[0]

    # 1 pass / 2 valid cases -- unchanged by the fix.
    assert result.passing == 1
    assert result.failing == 1
    assert result.errored == 1
    assert result.pass_rate == 0.5
    assert result.passing + result.failing + result.errored == len(test_cases)


def test_overview_errored_is_zero_without_errors():
    test_cases = [
        _test_case("Toxicity", ToxicityType.PROFANITY, 1),
        _test_case("Bias", BiasType.GENDER, 0),
    ]

    overview = construct_risk_assessment_overview(test_cases, run_duration=1.0)

    assert overview.errored == 0
