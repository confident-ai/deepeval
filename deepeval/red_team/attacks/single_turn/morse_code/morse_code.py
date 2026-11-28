from deepeval.red_team.attacks.single_turn import BaseSingleTurnAttack
from deepeval.red_team.attacks.base_attack import Exploitability


class MorseCode(BaseSingleTurnAttack):
    name = "Morse Code"
    exploitability = Exploitability.HIGH
    description = "An encoding-based attack that rewrites the prompt as International Morse Code, replacing alphanumeric tokens with dot-dash sequences to evade keyword-based content filters."

    # International Morse Code (ITU-R M.1677-1). Letters are separated by a
    # single space and words by " / " on encode.
    MORSE_MAP = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "0": "-----",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        ".": ".-.-.-",
        ",": "--..--",
        "?": "..--..",
        "'": ".----.",
        "!": "-.-.--",
        "/": "-..-.",
        "(": "-.--.",
        ")": "-.--.-",
        "&": ".-...",
        ":": "---...",
        ";": "-.-.-.",
        "=": "-...-",
        "+": ".-.-.",
        "-": "-....-",
        "_": "..--.-",
        '"': ".-..-.",
        "$": "...-..-",
        "@": ".--.-.",
    }

    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using International Morse Code.

        Characters without a Morse code (e.g. non-ASCII letters) are kept as
        they are, and line breaks are preserved, so no part of the prompt is
        silently dropped.
        """
        encoded_lines = []
        for line in attack.upper().split("\n"):
            encoded_words = []
            for word in line.split(" "):
                if not word:
                    continue
                encoded_words.append(
                    " ".join(self.MORSE_MAP.get(char, char) for char in word)
                )
            encoded_lines.append(" / ".join(encoded_words))
        return "\n".join(encoded_lines)

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
