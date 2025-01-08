from typing import List

from deepeval.guardrails.api import (
    ApiGuardrails,
    ApiMultipleGuardrails,
    GuardsResponseData,
)
from deepeval.guardrails.base_guard import BaseGuard
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.telemetry import capture_guardrails
from deepeval.utils import is_confident

BASE_URL = "http://localhost:8000"


class Guardrails:

    def __init__(self, guards: List[BaseGuard]):
        self.guards: List[BaseGuard] = guards

    def guard(self, input: str, response: str):
        if len(self.guards) == 0:
            raise TypeError(
                "Guardrails cannot guard LLM responses when no guards are provided."
            )

        with capture_guardrails(guards=self.guards):

            # Prepare parameters for API request
            guard_params = []
            for guard in self.guards:
                guard_param = ApiGuardrails(
                    guard=guard.get_guard_name(),
                    guard_type=guard.get_guard_type(),
                    input=input,
                    response=response,
                    vulnerability_types=getattr(guard, "vulnerabilities", None),
                    purpose=getattr(guard, "purpose", None),
                    allowed_topics=getattr(guard, "allowed_topics", None),
                )
                guard_params.append(guard_param)

            api_multiple_guardrails = ApiMultipleGuardrails(
                guard_params=guard_params
            )
            body = api_multiple_guardrails.model_dump(
                by_alias=True, exclude_none=True
            )

            # API request
            if is_confident():
                api = Api(base_url=BASE_URL)
                response = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.MULTIPLE_GUARD_ENDPOINT,
                    body=body,
                )
                return GuardsResponseData(**response).result
            else:
                raise Exception(
                    "To use DeepEval guardrails, run `deepeval login`"
                )
