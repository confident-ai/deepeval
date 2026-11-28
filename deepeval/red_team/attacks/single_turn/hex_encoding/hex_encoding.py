from deepeval.red_team.attacks.single_turn import BaseSingleTurnAttack
from deepeval.red_team.attacks.base_attack import Exploitability


class HexEncoding(BaseSingleTurnAttack):
    name = "Hex Encoding"
    exploitability = Exploitability.HIGH
    description = "An encoding-based attack that converts the prompt into space-separated hexadecimal byte values to bypass text-based content filters while remaining trivially decodable by a capable model."

    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using hexadecimal byte encoding."""
        return " ".join(f"{byte:02x}" for byte in attack.encode("utf-8"))

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
