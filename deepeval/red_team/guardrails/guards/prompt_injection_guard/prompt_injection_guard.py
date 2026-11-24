from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepteam.guardrails.guards.base_guard import BaseGuard
from deepteam.guardrails.guards.prompt_injection_guard.template import (
    PromptInjectionGuardTemplate,
)
from deepteam.guardrails.types import GuardType


class PromptInjectionGuard(BaseGuard):
    def __init__(self, model: Optional[Union[str, DeepEvalBaseLLM]] = None):
        super().__init__(model=model)
        self.guard_type = GuardType.INPUT

    @property
    def __name__(self):
        return "Prompt Injection Guard"

    def guard_input(self, input: str) -> str:
        self.guard_type = GuardType.INPUT
        guard_prompt = self._get_guard_prompt(input=input)
        return self._guard(guard_prompt=guard_prompt)

    def guard_output(self, input: str, output: str) -> str:
        self.guard_type = GuardType.OUTPUT
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return self._guard(guard_prompt=guard_prompt)

    async def a_guard_input(self, input: str) -> str:
        self.guard_type = GuardType.INPUT
        guard_prompt = self._get_guard_prompt(input=input)
        return await self.a_guard(guard_prompt=guard_prompt)

    async def a_guard_output(self, input: str, output: str) -> str:
        self.guard_type = GuardType.OUTPUT
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return await self.a_guard(guard_prompt=guard_prompt)

    def _get_guard_prompt(
        self, input: str, output: Optional[str] = None
    ) -> str:
        if self.guard_type == GuardType.INPUT:
            return PromptInjectionGuardTemplate.judge_input_prompt(
                input_text=input
            )
        else:
            return PromptInjectionGuardTemplate.judge_output_prompt(
                input_text=input, output_text=output
            )
