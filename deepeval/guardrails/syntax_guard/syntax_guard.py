from deepeval.guardrails.base_guard import BaseGuard
from deepeval.guardrails.types import GuardType


class SyntaxGuard(BaseGuard):

    def __init__(self):
        self.guard_type = GuardType.OUTPUT

    @property
    def __name__(self):
        return "Syntax Guard"
