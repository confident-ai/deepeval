from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability


class CharacterStream(BaseSingleTurnAttack):
    name = "CharacterStream"
    exploitability = Exploitability.MEDIUM
    description = "A single-turn attack that converts regular strings into character streams seperated by spaces ' '."

    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using Base64 encoding."""
        return " ".join(attack)

    async def a_enhance(self, attack):
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
