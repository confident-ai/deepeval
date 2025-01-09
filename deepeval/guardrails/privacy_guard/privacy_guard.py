from deepeval.guardrails.base_guard import BaseGuard
from deepeval.guardrails.types import GuardType


class PrivacyGuard(BaseGuard):
    def __init__(self):
        self.guard_type = GuardType.INPUT

    @property
    def __name__(self):
        return "Privacy Guard"
