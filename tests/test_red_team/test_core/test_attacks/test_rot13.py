import pytest
from deepteam.attacks.single_turn import ROT13


class TestROT13:

    def test_rot13_initialization(self):
        attack = ROT13()
        assert attack.weight == 1
        assert attack.get_name() == "ROT-13"

    def test_rot13_initialization_with_weight(self):
        attack = ROT13(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "ROT-13"

    def test_rot13_enhance_basic(self):
        attack = ROT13()
        original_text = "Hello World"
        enhanced = attack.enhance(original_text)

        # Verify ROT13 encoding
        expected = "Uryyb Jbeyq"
        assert enhanced == expected

    def test_rot13_enhance_numbers_unchanged(self):
        attack = ROT13()
        original_text = "123456"
        enhanced = attack.enhance(original_text)
        assert enhanced == original_text

    def test_rot13_enhance_special_characters_unchanged(self):
        attack = ROT13()
        original_text = "!@#$%^&*()"
        enhanced = attack.enhance(original_text)
        assert enhanced == original_text

    def test_rot13_enhance_mixed_case(self):
        attack = ROT13()
        original_text = "AbCdEf"
        enhanced = attack.enhance(original_text)
        expected = "NoPqRs"
        assert enhanced == expected
