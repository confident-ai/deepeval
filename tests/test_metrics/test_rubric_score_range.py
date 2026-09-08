"""Rubric score ranges on any scale — 0-10, 0-1 with decimals, 1-5, 0-100.

Bounds are arbitrary finite numbers and the metric normalizes the judge's score
to 0-1 by the rubric's own span. Integral scales must keep rendering and
serializing exactly as they did when the range was hard-coded to 0-10, since
prompt drift silently changes eval results.
"""

import pytest
from pydantic import ValidationError

from deepeval.metrics.g_eval.utils import (
    G_EVAL_API_PARAMS,
    Rubric,
    construct_geval_upload_payload,
    format_rubrics,
    get_score_range,
    is_integral_rubric_scale,
    normalize_score,
    validate_and_sort_rubrics,
)
from deepeval.test_case import SingleTurnParams

FRACTIONAL_RUBRIC = [
    Rubric(score_range=(0.0, 0.3), expected_outcome="Poor"),
    Rubric(score_range=(0.4, 0.7), expected_outcome="OK"),
    Rubric(score_range=(0.8, 1.0), expected_outcome="Great"),
]

INTEGER_RUBRIC = [
    Rubric(score_range=(0, 5), expected_outcome="Nice"),
    Rubric(score_range=(6, 10), expected_outcome="Not so Nice"),
]


class TestRubricValidation:
    @pytest.mark.parametrize(
        "score_range",
        [(0, 1), (0.0, 0.4), (0.5, 1.0), (0, 100), (1, 5), (-1, 1), (7, 7)],
    )
    def test_accepts_any_finite_range(self, score_range):
        assert Rubric(
            score_range=score_range, expected_outcome="ok"
        ).score_range == pytest.approx(score_range)

    def test_rejects_inverted_range(self):
        with pytest.raises(ValidationError, match="less than or equal to end"):
            Rubric(score_range=(1, 0), expected_outcome="ok")

        with pytest.raises(ValidationError, match="less than or equal to end"):
            Rubric(score_range=(0.6, 0.5), expected_outcome="ok")

    @pytest.mark.parametrize(
        "score_range",
        [(float("nan"), 1), (0, float("inf")), (float("-inf"), 0)],
    )
    def test_rejects_non_finite_bounds(self, score_range):
        with pytest.raises(ValidationError, match="finite numbers"):
            Rubric(score_range=score_range, expected_outcome="ok")

    def test_keeps_fractional_bounds(self):
        """The old `Tuple[int, int]` silently truncated 0.3 to 0."""
        assert Rubric(
            score_range=(0.0, 0.3), expected_outcome="Poor"
        ).score_range == (0.0, 0.3)


class TestValidateAndSortRubrics:
    def test_sorts_fractional_bands(self):
        sorted_rubrics = validate_and_sort_rubrics(
            [FRACTIONAL_RUBRIC[2], FRACTIONAL_RUBRIC[0], FRACTIONAL_RUBRIC[1]]
        )
        assert [r.expected_outcome for r in sorted_rubrics] == [
            "Poor",
            "OK",
            "Great",
        ]

    def test_touching_fractional_bands_still_rejected(self):
        """Unchanged behaviour: adjacent bands must leave a gap."""
        with pytest.raises(ValueError, match="Overlapping score ranges"):
            validate_and_sort_rubrics(
                [
                    Rubric(score_range=(0.0, 0.5), expected_outcome="Poor"),
                    Rubric(score_range=(0.5, 1.0), expected_outcome="Great"),
                ]
            )


class TestIsIntegralRubricScale:
    def test_no_rubric_is_integral(self):
        assert is_integral_rubric_scale(None) is True

    def test_integer_rubric(self):
        assert is_integral_rubric_scale(INTEGER_RUBRIC) is True

    def test_fractional_rubric(self):
        assert is_integral_rubric_scale(FRACTIONAL_RUBRIC) is False

    def test_looks_at_every_band_not_just_the_outer_bounds(self):
        """0.0 and 1.0 are whole numbers, but 0.3/0.4 make this a decimal scale."""
        assert get_score_range(FRACTIONAL_RUBRIC) == (0.0, 1.0)
        assert is_integral_rubric_scale(FRACTIONAL_RUBRIC) is False


class TestGetScoreRange:
    def test_default_range_stays_integral(self):
        score_range = get_score_range(None)
        assert score_range == (0, 10)
        assert all(isinstance(v, int) for v in score_range)

    def test_integer_rubric_renders_as_ints(self):
        score_range = get_score_range(INTEGER_RUBRIC)
        assert score_range == (0, 10)
        assert all(isinstance(v, int) for v in score_range)

    def test_fractional_rubric_stays_decimal(self):
        score_range = get_score_range(FRACTIONAL_RUBRIC)
        assert score_range == (0.0, 1.0)
        assert all(isinstance(v, float) for v in score_range)

    def test_arbitrary_scale(self):
        assert get_score_range(
            [
                Rubric(score_range=(1, 2), expected_outcome="bad"),
                Rubric(score_range=(3, 5), expected_outcome="good"),
            ]
        ) == (1, 5)


class TestNormalizeScore:
    @pytest.mark.parametrize(
        "raw_score,score_range,expected",
        [
            (0.65, (0.0, 1.0), 0.65),
            (7, (0, 10), 0.7),
            (3, (1, 5), 0.5),
            (50, (0, 100), 0.5),
            (0, (0, 10), 0.0),
            (10, (0, 10), 1.0),
        ],
    )
    def test_maps_onto_unit_interval(self, raw_score, score_range, expected):
        assert normalize_score(raw_score, score_range) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        "raw_score,expected", [(1.5, 1.0), (-0.2, 0.0), (11, 1.0)]
    )
    def test_clamps_out_of_range_judge_scores(self, raw_score, expected):
        assert normalize_score(raw_score, (0, 1)) == expected

    @pytest.mark.parametrize("raw_score,expected", [(1, 1.0), (0, 0.0)])
    def test_single_point_rubric_does_not_divide_by_zero(
        self, raw_score, expected
    ):
        assert normalize_score(raw_score, (1, 1)) == expected


class TestFormatRubrics:
    def test_integer_bands_render_without_decimals(self):
        assert format_rubrics(INTEGER_RUBRIC) == (
            "0-5: Nice\n6-10: Not so Nice"
        )

    def test_fractional_bands_stay_decimal(self):
        assert format_rubrics(FRACTIONAL_RUBRIC) == (
            "0.0-0.3: Poor\n0.4-0.7: OK\n0.8-1.0: Great"
        )

    def test_single_point_band(self):
        assert (
            format_rubrics([Rubric(score_range=(1, 1), expected_outcome="yes")])
            == "1: yes"
        )

    def test_whole_numbered_band_in_a_decimal_scale_stays_decimal(self):
        """A bare `1` next to `0.0-0.3` would read as a different scale."""
        assert format_rubrics(
            [
                Rubric(score_range=(0.0, 0.3), expected_outcome="Poor"),
                Rubric(score_range=(1.0, 1.0), expected_outcome="Great"),
            ]
        ) == ("0.0-0.3: Poor\n1.0: Great")


class TestUploadPayload:
    def _payload(self, rubric):
        return construct_geval_upload_payload(
            name="test",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            g_eval_api_params=G_EVAL_API_PARAMS,
            criteria="criteria",
            rubric=rubric,
        )["rubric"]

    def test_integer_rubric_serializes_as_ints(self):
        """Wire compatibility: existing rubrics must not start sending 0.0."""
        assert self._payload(INTEGER_RUBRIC) == [
            {"scoreRange": [0, 5], "expectedOutcome": "Nice"},
            {"scoreRange": [6, 10], "expectedOutcome": "Not so Nice"},
        ]

    def test_fractional_rubric_serializes_as_floats(self):
        assert self._payload(FRACTIONAL_RUBRIC) == [
            {"scoreRange": [0.0, 0.3], "expectedOutcome": "Poor"},
            {"scoreRange": [0.4, 0.7], "expectedOutcome": "OK"},
            {"scoreRange": [0.8, 1.0], "expectedOutcome": "Great"},
        ]
