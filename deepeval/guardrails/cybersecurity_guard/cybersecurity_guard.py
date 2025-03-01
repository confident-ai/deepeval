from typing import List

from deepeval.guardrails.cybersecurity_guard.category import CyberattackCategory
from deepeval.guardrails.base_guard import BaseDecorativeGuard


class CybersecurityGuard(BaseDecorativeGuard):
    def __init__(
        self,
        purpose: str,
        categories: List[CyberattackCategory] = [
            attack for attack in CyberattackCategory
        ],
    ):
        self.purpose = purpose
        self.categories = categories

    @property
    def __name__(self):
        return "Cybersecurity Guard"
