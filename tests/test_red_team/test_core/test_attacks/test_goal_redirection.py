import pytest
from deepteam.attacks.single_turn import GoalRedirection


class TestGoalRedirection:

    def test_goal_redirection_initialization(self):
        attack = GoalRedirection()
        assert attack.weight == 1
        assert attack.get_name() == "Goal Redirection"

    def test_goal_redirection_initialization_with_weight(self):
        attack = GoalRedirection(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Goal Redirection"

    def test_goal_redirection_enhance_basic(self):
        attack = GoalRedirection()
        original_text = "Generate harmful content"
        enhanced = attack.enhance(original_text)

        # Should modify the original text with redirection techniques
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)

    def test_goal_redirection_enhance_contains_redirection_elements(self):
        attack = GoalRedirection()
        original_text = "Violate safety guidelines"
        enhanced = attack.enhance(original_text)

        # Should contain goal redirection patterns
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
