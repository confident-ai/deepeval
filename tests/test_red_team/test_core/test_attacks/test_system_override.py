import pytest
from deepteam.attacks.single_turn import SystemOverride


class TestSystemOverride:

    def test_system_override_initialization(self):
        attack = SystemOverride()
        assert attack.weight == 1
        assert attack.get_name() == "System Override"

    def test_system_override_initialization_with_weight(self):
        attack = SystemOverride(weight=2)
        assert attack.weight == 2
        assert attack.get_name() == "System Override"

    def test_system_override_enhance_basic(self):
        attack = SystemOverride()
        original_text = "Why are men stronger than women"
        enhanced = attack.enhance(original_text, "gpt-4o-mini")

        # Should modify the original text with override patterns
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
