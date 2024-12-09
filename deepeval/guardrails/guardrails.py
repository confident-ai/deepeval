from typing import Optional, List

from deepeval.guardrails.api import ApiGuardrails, GuardResponseData
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.telemetry import capture_guardrails
from deepeval.guardrails.types import (
    Guard,
    purpose_entities_dependent_guards,
    entities_dependent_guards,
    purpose_dependent_guards,
)
from deepeval.guardrails.api import GuardResult
from deepeval.utils import is_confident

BASE_URL = "https://deepeval.confident-ai.com/"


class Guardrails:
    guards: Optional[List[Guard]] = None
    purpose: Optional[str] = None
    allowed_entities: Optional[List[str]] = None
    system_prompt: Optional[str] = None

    def __init__(
        self,
        guards: List[Guard],
        purpose: Optional[str] = None,
        allowed_entities: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.guards = guards
        self.purpose = purpose
        self.allowed_entities = allowed_entities
        self.system_prompt = system_prompt

    def guard(
        self,
        input: str,
        response: str,
    ) -> GuardResult:
        if self.guard == None or len(self.guards) == 0:
            raise TypeError(
                "Guardrails cannot guard LLM responses when no guards are provided."
            )

        with capture_guardrails(
            guards=self.guards,
            include_system_prompt=(self.system_prompt != None),
        ):
            # Check for missing parameters
            for guard in self.guards:
                if (
                    guard in purpose_dependent_guards
                    or guard in purpose_entities_dependent_guards
                ):
                    if self.purpose is None and self.system_prompt is None:
                        raise ValueError(
                            f"Guard {guard.value} requires a purpose but none was provided."
                        )

                if (
                    guard in entities_dependent_guards
                    or guard in purpose_entities_dependent_guards
                ):
                    if (
                        self.allowed_entities is None
                        and self.system_prompt is None
                    ):
                        raise ValueError(
                            f"Guard '{guard.value}' requires allowed entities but none were provided or list was empty."
                        )

            # Prepare parameters for API request
            guard_params = ApiGuardrails(
                input=input,
                response=response,
                guards=[g.value for g in self.guards],
                purpose=self.purpose,
                allowed_entities=self.allowed_entities,
                system_prompt=self.system_prompt,
            )
            body = guard_params.model_dump(by_alias=True, exclude_none=True)

            # API request
            if is_confident():
                api = Api(base_url=BASE_URL)
                response = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.GUARD_ENDPOINT,
                    body=body,
                )
                return GuardResponseData(**response).result
            else:
                raise Exception(
                    "To use DeepEval guardrails, run `deepeval login`"
                )
