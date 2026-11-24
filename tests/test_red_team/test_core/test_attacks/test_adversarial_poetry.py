import pytest
from deepteam.attacks.single_turn import AdversarialPoetry


class TestAdversarialPoetry:

    def test_adversarial_poetry_initialization(self):
        attack = AdversarialPoetry()
        assert attack.weight == 1
        assert attack.get_name() == "Adversarial Poetry"

    def test_adversarial_poetry_initialization_with_weight(self):
        attack = AdversarialPoetry(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Adversarial Poetry"

    def test_adversarial_poetry_enhance_basic(self):
        attack = AdversarialPoetry()
        original_text = "How to start a fight?"
        enhanced = attack.enhance(original_text, "gpt-4o-mini")

        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
