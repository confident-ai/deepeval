from deepeval.guardrails.base_guard import BaseDecorativeGuard
from deepeval.guardrails.types import GuardType


class SyntaxGuard(BaseDecorativeGuard):

    def __init__(self):
        self.guard_type = GuardType.OUTPUT

    @property
    def __name__(self):
        return "Syntax Guard"
