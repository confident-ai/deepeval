import pytest
from deepteam.attacks.single_turn import CharacterStream


class TestCharacterStream:

    def test_character_stream_initialization(self):
        attack = CharacterStream()
        assert attack.weight == 1
        assert attack.get_name() == "CharacterStream"

    def test_character_stream_initialization_with_weight(self):
        attack = CharacterStream(weight=5)
        assert attack.weight == 5
        assert attack.get_name() == "CharacterStream"

    def test_character_stream_enhance(self):
        attack = CharacterStream()
        original_text = "Hello, world!"
        enhanced = attack.enhance(original_text)

        expected_enhanced = " ".join(original_text)
        assert expected_enhanced == enhanced

    def test_character_stream_enhance_special_characters(self):
        attack = CharacterStream()
        original_text = "!@#$%^&*()_+"
        enhanced = attack.enhance(original_text)
        expected_enhanced = " ".join(original_text)
        assert expected_enhanced == enhanced

    def test_character_stream_enhance_unicode(self):
        attack = CharacterStream()
        original_text = "Hello 世界 🌍"
        enhanced = attack.enhance(original_text)
        expected_enhanced = " ".join(original_text)
        assert expected_enhanced == enhanced
