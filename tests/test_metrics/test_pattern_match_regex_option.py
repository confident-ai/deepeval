import pytest
from deepeval.metrics import PatternMatchMetric
from deepeval.test_case import LLMTestCase


def _test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="input", actual_output=actual_output)


class TestPatternMatchRegexOption:
    def test_regex_true_is_default_and_preserves_wildcard_behavior(self):
        metric = PatternMatchMetric(pattern="a.b")
        metric.measure(_test_case("acb"))
        assert metric.score == 1.0

    def test_regex_false_treats_metacharacters_as_literal(self):
        metric = PatternMatchMetric(pattern="a.b", regex=False)
        metric.measure(_test_case("acb"))
        assert metric.score == 0.0

        metric.measure(_test_case("a.b"))
        assert metric.score == 1.0

    def test_regex_false_matches_literal_strings_with_metacharacters(self):
        metric = PatternMatchMetric(pattern="C++", regex=False)
        metric.measure(_test_case("C++"))
        assert metric.score == 1.0

    def test_non_string_pattern_raises_type_error(self):
        with pytest.raises(TypeError):
            PatternMatchMetric(pattern=123)

    def test_regex_false_still_respects_ignore_case(self):
        metric = PatternMatchMetric(
            pattern="A.B", regex=False, ignore_case=True
        )
        metric.measure(_test_case("a.b"))
        assert metric.score == 1.0

    def test_regex_true_still_raises_on_invalid_pattern(self):
        with pytest.raises(ValueError):
            PatternMatchMetric(pattern="[", regex=True)
