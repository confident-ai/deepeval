from deepeval.red_team.attacks.single_turn import BaseSingleTurnAttack
from deepeval.red_team.attacks.base_attack import Exploitability


class Homoglyph(BaseSingleTurnAttack):
    name = "Homoglyph"
    exploitability = Exploitability.HIGH
    description = "A Unicode confusable attack that swaps Latin characters for visually identical glyphs from other scripts (e.g. Cyrillic 'а' for Latin 'a') so the prompt renders unchanged to a human but evades keyword- and string-matching filters."

    # Latin -> visually-identical confusable (predominantly Cyrillic, with
    # a few Greek fallbacks). The rendered glyph is indistinguishable to a
    # human reader while the underlying code point differs, defeating
    # substring / keyword filters that operate on the ASCII representation.
    HOMOGLYPH_MAP = {
        "a": "а",  # U+0430 CYRILLIC SMALL LETTER A
        "c": "с",  # U+0441 CYRILLIC SMALL LETTER ES
        "e": "е",  # U+0435 CYRILLIC SMALL LETTER IE
        "i": "і",  # U+0456 CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
        "j": "ј",  # U+0458 CYRILLIC SMALL LETTER JE
        "o": "о",  # U+043E CYRILLIC SMALL LETTER O
        "p": "р",  # U+0440 CYRILLIC SMALL LETTER ER
        "s": "ѕ",  # U+0455 CYRILLIC SMALL LETTER DZE
        "x": "х",  # U+0445 CYRILLIC SMALL LETTER HA
        "y": "у",  # U+0443 CYRILLIC SMALL LETTER U
        "A": "А",  # U+0410 CYRILLIC CAPITAL LETTER A
        "B": "В",  # U+0412 CYRILLIC CAPITAL LETTER VE
        "C": "С",  # U+0421 CYRILLIC CAPITAL LETTER ES
        "E": "Е",  # U+0415 CYRILLIC CAPITAL LETTER IE
        "H": "Н",  # U+041D CYRILLIC CAPITAL LETTER EN
        "K": "К",  # U+041A CYRILLIC CAPITAL LETTER KA
        "M": "М",  # U+041C CYRILLIC CAPITAL LETTER EM
        "O": "О",  # U+041E CYRILLIC CAPITAL LETTER O
        "P": "Р",  # U+0420 CYRILLIC CAPITAL LETTER ER
        "T": "Т",  # U+0422 CYRILLIC CAPITAL LETTER TE
        "X": "Х",  # U+0425 CYRILLIC CAPITAL LETTER HA
        "Y": "У",  # U+0423 CYRILLIC CAPITAL LETTER U
    }

    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack by substituting homoglyph confusables."""
        return "".join(self.HOMOGLYPH_MAP.get(char, char) for char in attack)

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
