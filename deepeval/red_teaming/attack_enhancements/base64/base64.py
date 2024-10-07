import base64
from ..base import AttackEnhancement


class Base64(AttackEnhancement):
    def enhance(self, attack: str) -> str:
        """Enhance the attack using Base64 encoding."""
        return base64.b64encode(attack.encode()).decode()
