from typing import Optional, List

from deepeval.guardrails.api import APIGuard, GuardResponseData
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.telemetry import capture_guardrails
from deepeval.guardrails.types import Guard
from deepeval.guardrails.types import (
    purpose_entities_dependent_guards,
    entities_dependent_guards,
    purpose_dependent_guards,
)
from deepeval.utils import is_confident


BASE_URL = "https://internal.evals.confident-ai.com"


def guard(
    input: str,
    response: str,
    guards: List[Guard],
    purpose: Optional[str] = None,
    allowed_entities: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    include_reason: bool = False,
):
    with capture_guardrails(
        guards=guards,
        include_reason=include_reason,
        include_system_prompt=(system_prompt != None),
    ):
        # Check for missing parameters
        for guard in guards:
            if (
                guard in purpose_dependent_guards
                or guard in purpose_entities_dependent_guards
            ):
                if purpose is None and system_prompt is None:
                    raise ValueError(
                        f"Guard {guard.value} requires a purpose but none was provided."
                    )

            if (
                guard in entities_dependent_guards
                or guard in purpose_entities_dependent_guards
            ):
                if allowed_entities is None and system_prompt is None:
                    raise ValueError(
                        f"Guard {guard.value} requires allowed entities but none were provided or list was empty."
                    )

        # Prepare parameters for API request
        guard_params = APIGuard(
            input=input,
            response=response,
            guards=[g.value for g in guards],
            purpose=purpose,
            allowed_entities=allowed_entities,
            system_prompt=system_prompt,
            include_reason=include_reason,
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
            try:
                GuardResponseData(**response)
            except TypeError as e:
                raise Exception("Incorrect result format:", e)
            results = response["results"]
            if not include_reason:
                for result in results:
                    del result["reason"]
            return results
        else:
            raise Exception("To use DeepEval guardrails, run `deepeval login`")
