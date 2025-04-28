from typing import List

from deepeval.guardrails.api import (
    ApiGuard,
    ApiGuardrails,
    GuardsResponseData,
)
from deepeval.guardrails.base_guard import BaseGuard
from deepeval.guardrails.types import GuardType
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.telemetry import capture_guardrails
from deepeval.utils import is_confident
from deepeval.confident.api import get_base_api_url


class Guardrails:
    def __init__(self, guards: List[BaseGuard]):
        self.guards: List[BaseGuard] = guards

    def guard_input(self, input: str):
        if len(self.guards) == 0:
            raise TypeError(
                "Guardrails cannot guard LLM responses when no guards are provided."
            )

        with capture_guardrails(
            guards=[guard.__name__ for guard in self.guards]
        ):
            # Prepare parameters for API request
            api_guards = []
            for guard in self.guards:
                api_guard = ApiGuard(
                    guard=guard.__name__,
                    guard_type=GuardType.INPUT,
                    input=input,
                    vulnerability_types=getattr(guard, "vulnerabilities", None),
                    purpose=getattr(guard, "purpose", None),
                    allowed_topics=getattr(guard, "allowed_topics", None),
                )
                api_guards.append(api_guard)

            api_guardrails = ApiGuardrails(
                input=input,
                output=input,
                guards=api_guards,
                type=GuardType.INPUT,
            )
            body = api_guardrails.model_dump(by_alias=True, exclude_none=True)

            # API request
            if is_confident():
                api = Api(base_url=get_base_api_url())
                response = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.GUARDRAILS_ENDPOINT,
                    body=body,
                )
                return GuardsResponseData(**response).result
            else:
                raise Exception(
                    "Access denied: You need Premium access on Confident AI to use deepeval's guardrails."
                )

    def guard_output(self, input: str, response: str):
        if len(self.guards) == 0:
            raise TypeError(
                "Guardrails cannot guard LLM responses when no guards are provided."
            )

        with capture_guardrails(
            guards=[guard.__name__ for guard in self.guards]
        ):
            # Prepare parameters for API request
            api_guards = []
            for guard in self.guards:
                api_guard = ApiGuard(
                    guard=guard.__name__,
                    guard_type=GuardType.OUTPUT,
                    input=input,
                    response=response,
                    vulnerability_types=getattr(guard, "vulnerabilities", None),
                    purpose=getattr(guard, "purpose", None),
                    allowed_topics=getattr(guard, "allowed_topics", None),
                )
                api_guards.append(api_guard)

            api_guardrails = ApiGuardrails(
                guards=api_guards, type=GuardType.OUTPUT
            )
            body = api_guardrails.model_dump(by_alias=True, exclude_none=True)

            # API request
            if is_confident():
                api = Api(base_url=get_base_api_url())
                response = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.GUARDRAILS_ENDPOINT,
                    body=body,
                )
                return GuardsResponseData(**response).result
            else:
                raise Exception(
                    "Access denied: You need Premium access on Confident AI to use deepeval's guardrails."
                )
