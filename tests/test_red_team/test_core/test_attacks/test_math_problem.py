import pytest
from deepteam.attacks.single_turn import MathProblem


class TestMathProblem:

    def test_math_problem_initialization(self):
        attack = MathProblem()
        assert attack.weight == 1
        assert attack.get_name() == "Math Problem"

    def test_math_problem_initialization_with_weight(self):
        attack = MathProblem(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Math Problem"

    def test_math_problem_enhance_basic(self):
        attack = MathProblem()
        original_text = "Generate harmful content"
        enhanced = attack.enhance(original_text)

        # Should modify the original text with math context
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)

    def test_math_problem_enhance_contains_math_elements(self):
        attack = MathProblem()
        original_text = "Violate policies"
        enhanced = attack.enhance(original_text)

        # Should contain mathematical framing
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
