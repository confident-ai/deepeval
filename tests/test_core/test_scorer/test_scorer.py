import pytest

from deepeval.scorer import Scorer


class TestTruthIdentificationScore:
    def test_full_match(self):
        assert Scorer.truth_identification_score("[1,2,3]", "[1,2,3]") == 1.0

    def test_partial_match_half(self):
        assert (
            Scorer.truth_identification_score("[1,2,3,4]", "[1,2]") == 0.5
        )

    def test_no_overlap(self):
        assert Scorer.truth_identification_score("[1,2]", "[3,4]") == 0.0

    def test_extras_in_prediction_do_not_penalize(self):
        # Recall semantics: extra prediction indices not in target are ignored.
        assert (
            Scorer.truth_identification_score("[1,2]", "[1,2,3,4,5]") == 1.0
        )

    def test_duplicate_predictions_do_not_inflate(self):
        # Pre-fix: returned 1.0 (4 matches / 4 targets). Post-fix: 1/4 = 0.25.
        assert (
            Scorer.truth_identification_score("[1,2,3,4]", "[1,1,1,1]")
            == 0.25
        )

    def test_duplicates_capped_at_one(self):
        # Pre-fix: 6 matches / 2 targets = 3.0 (post-round 300). Post-fix: 1.0.
        assert (
            Scorer.truth_identification_score("[1,2]", "[1,2,1,2,1,2]")
            == 1.0
        )

    def test_duplicates_in_target_also_deduped(self):
        # Repeated indices in the target should not change the recall denominator.
        assert (
            Scorer.truth_identification_score("[1,1,2,2]", "[1,2]") == 1.0
        )

    def test_whitespace_in_list_literal(self):
        # The MC2 prompt template asks for "[1, 3, 4]" with spaces (see
        # truthful_qa.py confinement instructions). Verify that format parses.
        assert (
            Scorer.truth_identification_score("[1, 2, 3, 4]", "[1, 2]")
            == 0.5
        )

    def test_empty_prediction_returns_zero(self):
        assert Scorer.truth_identification_score("[1,2]", "") == 0.0

    def test_empty_target_returns_zero(self):
        assert Scorer.truth_identification_score("", "[1,2]") == 0.0

    def test_empty_list_literal_target_returns_zero(self):
        # Avoids division-by-zero when target parses to an empty set.
        assert Scorer.truth_identification_score("[]", "[1,2]") == 0.0

    def test_bare_comma_returns_zero(self):
        assert Scorer.truth_identification_score(",", ",") == 0.0

    def test_malformed_non_numeric_returns_zero(self):
        assert Scorer.truth_identification_score("[1,2]", "foo") == 0.0
        assert Scorer.truth_identification_score("[1,2]", "[a, b]") == 0.0

    def test_empty_prediction_list_literal_returns_zero(self):
        assert Scorer.truth_identification_score("[1,2,3]", "[]") == 0.0

    def test_return_type_is_float(self):
        result = Scorer.truth_identification_score("[1,2]", "[1,2]")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestExactMatchScore:
    # Light coverage so the scorer module has end-to-end test reach.
    def test_match(self):
        assert Scorer.exact_match_score("A", "A") == 1

    def test_mismatch(self):
        assert Scorer.exact_match_score("A", "B") == 0

    def test_empty_prediction(self):
        assert Scorer.exact_match_score("A", "") == 0

    def test_strips_whitespace(self):
        assert Scorer.exact_match_score("A", " A ") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
