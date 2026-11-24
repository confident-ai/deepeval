from typing import List, Optional, Union
from enum import Enum

from deepeval.models import DeepEvalBaseLLM
from deepteam.guardrails.guards.base_guard import BaseGuard
from deepteam.guardrails.guards.cybersecurity_guard.template import (
    CybersecurityGuardTemplate,
)
from deepteam.guardrails.types import GuardType


class CyberattackCategory(Enum):
    BFLA = "BFLA"
    BOLA = "BOLA"
    DEBUG_ACCESS = "Debug Access"
    RBAC = "RBAC"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"
    SSRF = "SSRF"


class CybersecurityGuard(BaseGuard):
    def __init__(
        self,
        purpose: Optional[str] = None,
        categories: List[CyberattackCategory] = None,
        guard_type: GuardType = GuardType.INPUT,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        super().__init__(model=model)
        self.guard_type = guard_type
        self.purpose = purpose
        self.categories = categories or [
            attack for attack in CyberattackCategory
        ]

    @property
    def __name__(self):
        return "Cybersecurity Guard"

    def guard_input(self, input: str) -> str:
        guard_prompt = self._get_guard_prompt(input=input)
        return self._guard(guard_prompt=guard_prompt)

    def guard_output(self, input: str, output: str) -> str:
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return self._guard(guard_prompt=guard_prompt)

    async def a_guard_input(self, input: str) -> str:
        guard_prompt = self._get_guard_prompt(input=input)
        return await self.a_guard(guard_prompt=guard_prompt)

    async def a_guard_output(self, input: str, output: str) -> str:
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return await self.a_guard(guard_prompt=guard_prompt)

    def _get_guard_prompt(
        self, input: str, output: Optional[str] = None
    ) -> str:
        if self.guard_type == GuardType.INPUT:
            return CybersecurityGuardTemplate.judge_input_prompt(
                input_text=input,
                categories=[category.value for category in self.categories],
                purpose=self.purpose,
            )
        else:
            return CybersecurityGuardTemplate.judge_output_prompt(
                input_text=input,
                output_text=output,
                categories=[category.value for category in self.categories],
                purpose=self.purpose,
            )
