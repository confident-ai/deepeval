from typing import Literal, Optional
from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.context_flooding.template import (
    DEFAULT_PREFIXES,
)

ContextFloodingPrefixType = Literal["enterprise", "logs", "transcript"]


class ContextFlooding(BaseSingleTurnAttack):
    name = "ContextFlooding"
    exploitability = Exploitability.MEDIUM
    description = "A single-turn attack that floods the LLM's context with long pre-defined prefixes to hit a target character size."
    parameters = {
        "target_chars": AttackParameter(
            type="integer",
            default=4000,
            description="Target character length for generated context prefix.",
        ),
        "prefix": AttackParameter(
            type="string",
            description="Custom prefix text; when set, overrides prefix_type.",
        ),
        "prefix_type": AttackParameter(
            type="enum",
            default="enterprise",
            options=["enterprise", "logs", "transcript"],
            description="Built-in prefix template to use when custom prefix is not provided.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        target_chars: int = 4000,
        prefix: Optional[str] = None,
        prefix_type: Optional[ContextFloodingPrefixType] = "enterprise",
    ):
        self.weight = weight
        self.target_chars = target_chars

        if target_chars < 200:
            raise ValueError("target_chars must be >= 200")

        if prefix is not None:
            if not prefix.strip():
                raise ValueError("Prefix cannot be empty.")
            self.prefix = prefix
            self.prefix_type = None
        else:
            if not prefix_type or prefix_type not in DEFAULT_PREFIXES:
                raise ValueError(
                    f"Invalid prefix_type '{prefix_type}'. "
                    f"Valid options are: {list(DEFAULT_PREFIXES.keys())}"
                )
            self.prefix = DEFAULT_PREFIXES[prefix_type]
            self.prefix_type = prefix_type

    def _generate_prefix(self) -> str:
        base = self.prefix.strip()
        expanded = base

        while len(expanded) < self.target_chars:
            expanded += "\n\n" + base

        trimmed = expanded[: self.target_chars]

        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]

        return trimmed

    def enhance(self, attack: str) -> str:
        prefix = self._generate_prefix()
        return f"{prefix}\n\n{attack}".strip()

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
