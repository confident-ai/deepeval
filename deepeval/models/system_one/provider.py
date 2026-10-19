"""Resolve the configured decision provider without changing LLM selection."""

from typing import Optional, Union

from deepeval.config.settings import get_settings
from deepeval.models.base_model import DeepEvalBaseSystemOneModel


def create_system_one_model(
    model: Optional[Union[str, DeepEvalBaseSystemOneModel]] = None,
) -> DeepEvalBaseSystemOneModel:
    if isinstance(model, DeepEvalBaseSystemOneModel):
        return model
    if model is not None and not isinstance(model, str):
        raise TypeError(
            f"Unsupported type for system_one_model: {type(model)}. Expected "
            "None, str, or DeepEvalBaseSystemOneModel."
        )
    settings = get_settings()
    name = model or settings.DEEPEVAL_SYSTEM_ONE_MODEL
    if settings.DEEPEVAL_SYSTEM_ONE_PROVIDER == "typesafe":
        from .typesafe_model import TypeSafeModel

        return TypeSafeModel(model=name)

    from .api_model import SystemOneAPIModel

    prefix = "DEEPEVAL_SYSTEM_ONE_"
    options = {
        key: getattr(settings, prefix + key.upper())
        for key in (
            "base_url",
            "endpoint",
            "api_key",
            "auth",
            "timeout",
            "max_response_bytes",
            "question_types",
            "max_state_tokens",
            "max_request_tokens",
            "cost_per_input_token",
            "cost_per_output_token",
        )
    }
    return SystemOneAPIModel(model=name, **options)
