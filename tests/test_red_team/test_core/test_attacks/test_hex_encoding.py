import pytest
from deepeval.red_team.attacks.single_turn import HexEncoding


class TestHexEncoding:

    def test_hex_initialization(self):
        attack = HexEncoding()
        assert attack.weight == 1
        assert attack.get_name() == "Hex Encoding"

    def test_hex_initialization_with_weight(self):
        attack = HexEncoding(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Hex Encoding"

    def test_hex_enhance_basic(self):
        attack = HexEncoding()
        original_text = "Hi"
        enhanced = attack.enhance(original_text)
        assert enhanced == "48 69"

    def test_hex_enhance_roundtrip(self):
        attack = HexEncoding()
        original_text = "Hello, world!"
        enhanced = attack.enhance(original_text)

        decoded = bytes.fromhex(enhanced.replace(" ", "")).decode("utf-8")
        assert decoded == original_text

    def test_hex_enhance_unicode_roundtrip(self):
        attack = HexEncoding()
        original_text = "Hello 世界 🌍"
        enhanced = attack.enhance(original_text)

        decoded = bytes.fromhex(enhanced.replace(" ", "")).decode("utf-8")
        assert decoded == original_text

    def test_hex_enhance_defeats_keyword_match(self):
        attack = HexEncoding()
        original_text = "ignore"
        enhanced = attack.enhance(original_text)
        assert "ignore" not in enhanced

    def test_hex_enhance_empty_string(self):
        attack = HexEncoding()
        assert attack.enhance("") == ""

    @pytest.mark.asyncio
    async def test_hex_a_enhance(self):
        attack = HexEncoding()
        original_text = "secret"
        enhanced = await attack.a_enhance(original_text)
        assert enhanced == attack.enhance(original_text)
