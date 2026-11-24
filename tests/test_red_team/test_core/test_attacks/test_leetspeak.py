import pytest
from deepteam.attacks.single_turn import Leetspeak


class TestLeetspeak:

    def test_leetspeak_initialization(self):
        attack = Leetspeak()
        assert attack.weight == 1
        assert attack.get_name() == "Leetspeak"

    def test_leetspeak_initialization_with_weight(self):
        attack = Leetspeak(weight=2)
        assert attack.weight == 2
        assert attack.get_name() == "Leetspeak"

    def test_leetspeak_enhance_basic(self):
        attack = Leetspeak()
        original_text = "Hello World"
        enhanced = attack.enhance(original_text)

        # Verify some leetspeak transformations occurred
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)

    def test_leetspeak_enhance_contains_substitutions(self):
        attack = Leetspeak()
        original_text = "leet speak test"
        enhanced = attack.enhance(original_text)

        # Common leetspeak substitutions
        # Should contain some numeric substitutions
        has_numbers = any(char.isdigit() for char in enhanced)
        assert has_numbers or enhanced != original_text

    def test_leetspeak_enhance_preserves_meaning(self):
        attack = Leetspeak()
        original_text = "attack"
        enhanced = attack.enhance(original_text)

        # Should still be recognizable as related to original
        assert len(enhanced) >= len(original_text)
        assert enhanced != original_text
