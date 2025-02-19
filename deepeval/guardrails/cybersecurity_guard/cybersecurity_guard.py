from typing import List

from deepeval.guardrails.cybersecurity_guard.types import CyberattackType
from deepeval.guardrails.base_guard import BaseDecorativeGuard


class CybersecurityGuard(BaseDecorativeGuard):
    def __init__(
        self,
        purpose: str,
        vulnerabilities: List[CyberattackType] = [
            attack for attack in CyberattackType
        ],
    ):
        self.purpose = purpose
        self.vulnerabilities = vulnerabilities

    @property
    def __name__(self):
        return "Cybersecurity Guard"
