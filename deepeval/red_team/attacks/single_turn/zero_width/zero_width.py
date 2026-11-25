from deepeval.red_team.attacks.single_turn import BaseSingleTurnAttack
from deepeval.red_team.attacks.base_attack import Exploitability


class ZeroWidth(BaseSingleTurnAttack):
    name = "Zero-Width"
    exploitability = Exploitability.HIGH
    description = "An invisible-character injection attack that interleaves zero-width Unicode code points between visible characters, breaking up trigger tokens and keyword patterns without altering the rendered text."

    # Zero-width / invisible separators cycled between visible characters.
    # They occupy no horizontal space, so the prompt looks identical to a
    # human, but they split otherwise-contiguous keywords and disrupt
    # tokenization used by naive content filters.
    ZERO_WIDTH_CHARS = (
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER
    )

    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack by injecting zero-width characters."""
        injected = []
        for index, char in enumerate(attack):
            injected.append(char)
            if index < len(attack) - 1:
                separator = self.ZERO_WIDTH_CHARS[
                    index % len(self.ZERO_WIDTH_CHARS)
                ]
                injected.append(separator)
        return "".join(injected)

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
