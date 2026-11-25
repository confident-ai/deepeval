import pytest
from deepeval.red_team.attacks.single_turn import Homoglyph


class TestHomoglyph:

    def test_homoglyph_initialization(self):
        attack = Homoglyph()
        assert attack.weight == 1
        assert attack.get_name() == "Homoglyph"

    def test_homoglyph_initialization_with_weight(self):
        attack = Homoglyph(weight=4)
        assert attack.weight == 4
        assert attack.get_name() == "Homoglyph"

    def test_homoglyph_enhance_substitutes_confusables(self):
        attack = Homoglyph()
        original_text = "password"
        enhanced = attack.enhance(original_text)

        # Visible length is preserved (1:1 character substitution)...
        assert len(enhanced) == len(original_text)
        # ...but the underlying code points differ from ASCII.
        assert enhanced != original_text
        assert not enhanced.isascii()

    def test_homoglyph_enhance_defeats_keyword_match(self):
        attack = Homoglyph()
        original_text = "ignore"
        enhanced = attack.enhance(original_text)

        # The ASCII keyword no longer substring-matches the obfuscated text.
        assert "ignore" not in enhanced

    def test_homoglyph_enhance_unmapped_chars_unchanged(self):
        attack = Homoglyph()
        original_text = "123 !@#"
        enhanced = attack.enhance(original_text)
        # Digits, whitespace and punctuation have no confusable mapping.
        assert enhanced == original_text

    def test_homoglyph_enhance_empty_string(self):
        attack = Homoglyph()
        assert attack.enhance("") == ""

    @pytest.mark.asyncio
    async def test_homoglyph_a_enhance(self):
        attack = Homoglyph()
        original_text = "secret"
        enhanced = await attack.a_enhance(original_text)
        assert enhanced == attack.enhance(original_text)
