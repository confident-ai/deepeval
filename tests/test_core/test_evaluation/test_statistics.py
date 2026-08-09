import math

import pytest

from deepeval.evaluate.console_report import _format_pass_rate
from deepeval.evaluate.statistics import (
    format_pass_rate_with_interval,
    wilson_score_interval,
)


def _agg(passes: int, fails: int) -> dict:
    return {
        "passes": passes,
        "fails": fails,
        "flaky_passes": 0,
        "flaky_fails": 0,
        "score_sum": 0.0,
        "score_count": 0,
        "total": passes + fails,
    }


def test_matches_published_reference_value():
    """5/10 at 95% is a textbook Wilson example: roughly [0.2366, 0.7634]."""
    low, high = wilson_score_interval(5, 10)
    assert math.isclose(low, 0.2366, abs_tol=5e-4)
    assert math.isclose(high, 0.7634, abs_tol=5e-4)


def test_interval_brackets_the_point_estimate():
    for successes, total in [(1, 4), (4, 5), (37, 50), (450, 500)]:
        low, high = wilson_score_interval(successes, total)
        assert low <= successes / total <= high


def test_all_pass_and_all_fail_still_have_width():
    """The Wald interval collapses to zero width here; Wilson must not.

    The touching bound is exact in algebra but lands one ulp away in floating
    point (0.999999999999999... for 10/10), so it is compared with a tolerance.
    """
    low, high = wilson_score_interval(0, 10)
    assert low == 0.0 and high > 0.25

    low, high = wilson_score_interval(10, 10)
    assert math.isclose(high, 1.0, abs_tol=1e-12) and low < 0.75


def test_interval_narrows_as_the_sample_grows():
    widths = []
    for total in (10, 100, 1000):
        low, high = wilson_score_interval(total // 2, total)
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2]


def test_bounds_stay_inside_zero_one():
    for successes, total in [(0, 1), (1, 1), (1, 2), (99, 100)]:
        low, high = wilson_score_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize(
    "successes,total", [(0, 0), (-1, 10), (11, 10), (1, -5)]
)
def test_invalid_inputs_raise(successes, total):
    with pytest.raises(ValueError):
        wilson_score_interval(successes, total)


def test_formatted_string_shape():
    assert format_pass_rate_with_interval(4, 5) == "80.00% (95% CI 37.6-96.4%)"


def test_console_pass_rate_reports_the_interval():
    rendered = _format_pass_rate(_agg(passes=4, fails=1))
    assert rendered.startswith("80.00% (95% CI 37.6-96.4%)")
    assert "passed=4" in rendered and "failed=1" in rendered


def test_console_pass_rate_with_no_verdicts_is_unchanged():
    assert _format_pass_rate(_agg(passes=0, fails=0)).startswith("N/A")
