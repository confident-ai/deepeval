from typing import List, Optional

from deepeval.guardrails.cybersecurity_guard.types import CyberattackType
from deepeval.guardrails.api import ApiGuardrails, GuardResponseData
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.guardrails.base_guard import BaseGuard
from deepeval.guardrails.types import GuardType
from deepeval.guardrails.api import BASE_URL
from deepeval.utils import is_confident


class CyberSecurityGuard(BaseGuard):

    def __init__(
        self, 
        purpose: str,
        guard_type: GuardType = GuardType.INPUT,
        cyberattack_types: List[CyberattackType] = [
            attack for attack in CyberattackType
        ]
    ):
        self.purpose = purpose
        self.guard_type = guard_type
        self.cyberattack_types =cyberattack_types

    def guard(
        self, 
        input: Optional[str] = None, 
        response: Optional[str] = None
    ) -> int:
        guard_params = ApiGuardrails(
            guard=self.get_guard_name(),
            guard_type=self.guard_type.value,
            input=input,
            response=response,
            purpose=self.purpose,
            vulnerability_types=[attack.value for attack in self.cyberattack_types]
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
        
    def get_guard_name(self) -> str:
        if self.guard_type == GuardType.INPUT:
            return "Cybersecurity Input Guard"
        elif self.guard_type == GuardType.OUTPUT:
            return "Cybersecurity Output Guard"