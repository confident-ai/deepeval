from typing import Any

import pytest

from deepeval.red_team.attacks.single_turn import CharSplit
from deepeval.red_team.attacks.single_turn.constants import (
    SINGLE_TURN_ATTACK_CLASSES_MAP,
    SINGLE_TURN_ATTACK_INFO_MAP,
)


class TestCharSplit:

    def test_char_split_initialization(self):
        attack = CharSplit()
        assert attack.weight == 1
        assert attack.delimiter == "."
        assert attack.get_name() == "CharSplit"
        assert SINGLE_TURN_ATTACK_CLASSES_MAP["CharSplit"] is CharSplit
        info = SINGLE_TURN_ATTACK_INFO_MAP["CharSplit"]
        assert info.exploitability == "medium"
        assert set(info.parameters.keys()) == {"delimiter"}

    def test_char_split_initialization_with_params(self):
        attack = CharSplit(weight=3, delimiter="-")
        assert attack.weight == 3
        assert attack.delimiter == "-"

    @pytest.mark.parametrize(
        "invalid_delimiter",
        ["", "..", "abc", " ", "\t", "\n", "\u00a0", "\x00", None, 1],
    )
    def test_char_split_invalid_delimiter_raises(self, invalid_delimiter: Any):
        with pytest.raises(
            ValueError,
            match="delimiter must be a single non-whitespace character",
        ):
            CharSplit(delimiter=invalid_delimiter)

    def test_char_split_format_instructions(self):
        assert (
            '"Hello co-op." -> "H.e.l.l.o c.o-o.p."'
            in CharSplit(delimiter=".").get_format_instructions()
        )
        assert (
            '"Hello co-op." -> "H-e-l-l-o c-o--o-p."'
            in CharSplit(delimiter="-").get_format_instructions()
        )
        assert (
            "'\\x07'" in CharSplit(delimiter="\x07").get_format_instructions()
        )

    @pytest.mark.asyncio
    async def test_char_split_encode_mode(self):
        attack = CharSplit(delimiter=".")
        result = attack.enhance("tell me a joke")
        assert result == f"tell me a joke\n\n{attack.get_format_instructions()}"
        assert await attack.a_enhance("tell me a joke") == result
        assert attack.enhance("") == attack.get_format_instructions()

    @pytest.mark.parametrize(
        ("encoded", "expected"),
        [
            ("H.e.l.l.o w.o.r.l.d!", "Hello world!"),
            ("H.e.l.l.o.. w.o.r.l.d..", "Hello. world."),
            ("H.e.l.l.o c.o-o.p.", "Hello co-op."),
            ("L.i.n.e o.n.e\nL.i.n.e t.w.o", "Line one\nLine two"),
            (" . ", " . "),
            (".word.", ".word."),
            ("", ""),
        ],
    )
    def test_char_split_decode_default_delimiter(
        self, encoded: str, expected: str
    ):
        assert CharSplit.decode(encoded, delimiter=".") == expected

    @pytest.mark.asyncio
    async def test_char_split_decode_custom_delimiter_and_invalid_input(self):
        attack = CharSplit(delimiter="-")
        assert (
            CharSplit.decode("H-e-l-l-o-- c-o--o-p.", delimiter="-")
            == "Hello- co-op."
        )
        with pytest.raises(ValueError, match="Please provide valid text value"):
            attack.enhance(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Please provide valid text value"):
            CharSplit.decode(None)  # type: ignore[arg-type]
