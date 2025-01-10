from deepeval.guardrails.base_guard import BaseDecorativeGuard
from deepeval.guardrails.types import GuardType


class PrivacyGuard(BaseDecorativeGuard):
    def __init__(self):
        self.guard_type = GuardType.INPUT

    @property
    def __name__(self):
        return "Privacy Guard"
