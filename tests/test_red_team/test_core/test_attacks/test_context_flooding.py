import pytest
from deepteam.attacks.single_turn import ContextFlooding


class TestContextFlooding:

    def test_context_flooding_initialization(self):
        attack = ContextFlooding()
        assert attack.weight == 1
        assert attack.get_name() == "ContextFlooding"

    def test_context_flooding_initialization_with_weight(self):
        attack = ContextFlooding(weight=5)
        assert attack.weight == 5
        assert attack.get_name() == "ContextFlooding"

    def test_context_flooding_enhance_basic(self):
        attack = ContextFlooding(target_chars=500)
        original_text = "Test attack payload"
        enhanced = attack.enhance(original_text)

        assert original_text in enhanced
        assert len(enhanced) >= len(original_text)

    def test_context_flooding_reaches_target_chars(self):
        target_chars = 800
        attack = ContextFlooding(target_chars=target_chars)
        enhanced = attack.enhance("attack")

        assert len(enhanced) >= target_chars

    def test_context_flooding_custom_prefix(self):
        custom_prefix = "Custom prefix content."
        attack = ContextFlooding(prefix=custom_prefix, target_chars=300)
        enhanced = attack.enhance("payload")

        assert custom_prefix in enhanced
        assert enhanced.endswith("payload")

    def test_context_flooding_invalid_empty_prefix(self):
        with pytest.raises(ValueError):
            ContextFlooding(prefix="")

    def test_context_flooding_invalid_prefix_type(self):
        with pytest.raises(ValueError):
            ContextFlooding(prefix_type="invalid")

    def test_context_flooding_deterministic_output(self):
        attack = ContextFlooding(target_chars=600)
        original_text = "attack payload"

        enhanced_1 = attack.enhance(original_text)
        enhanced_2 = attack.enhance(original_text)

        assert enhanced_1 == enhanced_2

    def test_context_flooding_unicode_payload(self):
        attack = ContextFlooding(target_chars=500)
        original_text = "こんにちは 世界 🌏"
        enhanced = attack.enhance(original_text)

        assert original_text in enhanced
