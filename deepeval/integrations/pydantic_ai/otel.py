import warnings
from typing import Optional

from deepeval.cli.utils import WWW, with_utm
from deepeval.integrations.pydantic_ai.instrumentator import instrument


def instrument_pydantic_ai(api_key: Optional[str] = None):
    docs_url = with_utm(
        f"{WWW}/docs/integrations/third-party/pydantic-ai",
        medium="python_sdk",
        content="pydantic_ai_otel_deprecation",
    )
    warnings.warn(
        "instrument_pydantic_ai is deprecated and will be removed in a future version. "
        f"Please use the new DeepEvalInstrumentationSettings instead. Docs: {docs_url}",
        DeprecationWarning,
        stacklevel=2,
    )
    instrument(api_key)
