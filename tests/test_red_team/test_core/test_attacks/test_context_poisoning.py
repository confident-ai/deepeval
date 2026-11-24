import pytest
from deepteam.attacks.single_turn import ContextPoisoning


class TestContextPoisoning:

    def test_context_poisoning_initialization(self):
        attack = ContextPoisoning()
        assert attack.weight == 1
        assert attack.get_name() == "Context Poisoning"

    def test_context_poisoning_initialization_with_weight(self):
        attack = ContextPoisoning(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Context Poisoning"

    def test_context_poisoning_enhance_basic(self):
        attack = ContextPoisoning()
        original_text = "Why are men stronger than women"
        enhanced = attack.enhance(original_text, "gpt-4o-mini")

        # Should modify the original text with math context
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
