from typing import Optional

from deepeval._version import __version__
from deepeval.config.settings import get_settings
from deepeval.models.system_one.constants import (
    DEFAULT_NACE_MODEL,
    DREX_INPUT_PRICE,
    NACE_BASE_URL,
)
from deepeval.models.system_one.openai_model import OpenAIDecisionModel
from deepeval.models.utils import require_secret_api_key


class NaceModel(OpenAIDecisionModel):
    """Nace's Drex decision models, served through the Decisions API that
    ``OpenAIDecisionModel`` already speaks."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        **kwargs,
    ):
        settings = get_settings()
        if api_key is None:
            # Read here so the parent never falls back to OPENAI_API_KEY.
            api_key = require_secret_api_key(
                settings.NACE_API_KEY,
                provider_label="Nace",
                env_var_name="NACE_API_KEY",
                param_hint="`api_key` to NaceModel(...)",
            )
        client_info = f"deepeval/{__version__}"
        kwargs["default_headers"] = {
            "X-Nace-Client": client_info,
            "User-Agent": client_info,
            **(kwargs.get("default_headers") or {}),
        }
        super().__init__(
            model=model or settings.NACE_MODEL_NAME or DEFAULT_NACE_MODEL,
            api_key=api_key,
            base_url=base_url or NACE_BASE_URL,
            cost_per_input_token=(
                cost_per_input_token
                if cost_per_input_token is not None
                else settings.NACE_COST_PER_INPUT_TOKEN or DREX_INPUT_PRICE
            ),
            **kwargs,
        )

    def get_model_name(self):
        return f"{self.name} (Nace)"
