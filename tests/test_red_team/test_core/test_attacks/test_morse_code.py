import pytest
from deepeval.red_team.attacks.single_turn import MorseCode


class TestMorseCode:

    def test_morse_initialization(self):
        attack = MorseCode()
        assert attack.weight == 1
        assert attack.get_name() == "Morse Code"

    def test_morse_initialization_with_weight(self):
        attack = MorseCode(weight=2)
        assert attack.weight == 2
        assert attack.get_name() == "Morse Code"

    def test_morse_enhance_basic(self):
        attack = MorseCode()
        original_text = "SOS"
        enhanced = attack.enhance(original_text)
        assert enhanced == "... --- ..."

    def test_morse_enhance_word_separator(self):
        attack = MorseCode()
        original_text = "HI ME"
        enhanced = attack.enhance(original_text)
        # Letters separated by space, words by " / ".
        assert enhanced == ".... .. / -- ."

    def test_morse_enhance_case_insensitive(self):
        attack = MorseCode()
        assert attack.enhance("sos") == attack.enhance("SOS")

    def test_morse_enhance_defeats_keyword_match(self):
        attack = MorseCode()
        original_text = "ignore"
        enhanced = attack.enhance(original_text)
        assert "ignore" not in enhanced

    def test_morse_enhance_digits(self):
        attack = MorseCode()
        assert attack.enhance("1") == ".----"

    def test_morse_enhance_unmapped_chars_do_not_raise(self):
        # Punctuation, apostrophes, symbols and non-ASCII must never raise a
        # KeyError mid-engagement; unmapped characters are kept as they are.
        attack = MorseCode()
        for text in ["Don't do it!", "call 911 now.", "a@b#c$ 42%", "héllo"]:
            assert isinstance(attack.enhance(text), str)
        assert attack.enhance("héllo") == ".... É .-.. .-.. ---"

    def test_morse_enhance_preserves_line_breaks(self):
        attack = MorseCode()
        assert attack.enhance("hi\nme") == ".... ..\n-- ."

    def test_morse_enhance_empty_string(self):
        attack = MorseCode()
        assert attack.enhance("") == ""

    @pytest.mark.asyncio
    async def test_morse_a_enhance(self):
        attack = MorseCode()
        original_text = "secret"
        enhanced = await attack.a_enhance(original_text)
        assert enhanced == attack.enhance(original_text)
