from typing import List

from deepeval.guardrails.cybersecurity_guard.types import CyberattackType
from deepeval.guardrails.base_guard import BaseDecorativeGuard
from deepeval.guardrails.types import GuardType


class CybersecurityGuard(BaseDecorativeGuard):
    def __init__(
        self,
        purpose: str,
        guard_type: GuardType = GuardType.INPUT,
        vulnerabilities: List[CyberattackType] = [
            attack for attack in CyberattackType
        ],
    ):
        self.purpose = purpose
        self.guard_type = guard_type
        self.vulnerabilities = vulnerabilities

    @property
    def __name__(self):
        "Cybersecurity Guard"
