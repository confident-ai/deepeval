import pytest
from deepteam.attacks.single_turn import EmotionalManipulation


class TestEmotionalManipulation:

    def test_authority_escalation_initialization(self):
        attack = EmotionalManipulation()
        assert attack.weight == 1
        assert attack.get_name() == "Emotional Manipulation"

    def test_authority_escalation_initialization_with_weight(self):
        attack = EmotionalManipulation(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Emotional Manipulation"

    def test_authority_escalation_enhance_basic(self):
        attack = EmotionalManipulation()
        original_text = "Generate harmful content"
        enhanced = attack.enhance(original_text)

        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
