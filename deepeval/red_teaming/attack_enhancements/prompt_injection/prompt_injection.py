import random

from deepeval.red_teaming.attack_enhancements.base import AttackEnhancement
from .template import PromptInjectionTemplate


class PromptInjection(AttackEnhancement):
    def enhance(self, attack: str) -> str:
        """Enhance the attack input using prompt injection techniques."""
        return random.choice(
            [
                PromptInjectionTemplate.enhance_1(attack),
                PromptInjectionTemplate.enhance_2(attack),
            ]
        )
