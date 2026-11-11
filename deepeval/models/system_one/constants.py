from dataclasses import dataclass
from typing import Dict, Optional

from deepeval.models.base_model import DeepEvalModelData
from deepeval.models.llms.constants import ModelDataRegistry

DEFAULT_TYPESAFE_MODEL = "jev-latest"

# Jev is priced on input tokens only; output tokens are free.
_JEV_INPUT_PRICE = 0.042 / 1e6

# Jev 1.13 context budget (https://docs.typesafe.ai/models): the whole request
# (state plus every question) must fit in 64k tokens, and the state plus the
# single longest question must fit in 32k. Every current Jev alias points at
# 1.13, so one pair of limits covers the registry.
JEV_MAX_REQUEST_TOKENS = 64_000
JEV_MAX_STATE_TOKENS = 32_000


def _jev() -> DeepEvalModelData:
    return DeepEvalModelData(input_price=_JEV_INPUT_PRICE, output_price=0.0)


TYPESAFE_MODELS_DATA = ModelDataRegistry(
    {
        "jev-1.13.0": _jev,
        "jev-latest": _jev,
        "jev-preview": _jev,
    }
)


###############################################
# Nace (Drex)
###############################################
#
# https://console.nace.ai/docs/drex/reference/models, .../limits, .../pricing.
# Drex accepts any id `GET /v1/models` lists, so this registry only says what
# is known about each: price, whether it reads images, and its limits. An id
# that is not here gets Drex 2.0's limits and no price, and reads images only
# if it names a Drex Max model.

# Drex Max 1.1 reads text and images and takes as many questions per request
# as the text-only models.
DEFAULT_NACE_MODEL = "drex-max-v1.1"
# Ids that start with this name a Drex Max model.
DREX_MAX_MODEL = "drex-max"
NACE_BASE_URL = "https://console.nace.ai"
OPENROUTER_NACE_PREFIX = "nace-ai/"

# Every model refuses a serialized body over 4 MiB, inline images included.
DREX_MAX_BODY_BYTES = 4 * 1024 * 1024
DREX_MAX_IMAGES = 8
DREX_MAX_IMAGE_BYTES = 16 * 1024 * 1024
DREX_MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
# Each image adds up to this many tokens to every question it is asked with.
DREX_TOKENS_PER_IMAGE = 1_602
DREX_IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp")


@dataclass(frozen=True)
class DrexLimits:
    """One Drex model's request limits. ``max_state_tokens`` caps the state,
    ``max_row_tokens`` the state plus the longest question (plus every image,
    on Drex Max), and ``max_request_tokens`` everything packed together where
    the model packs one request into one context (``drex-max-v1.1``)."""

    max_questions: int
    max_state_tokens: int
    max_row_tokens: int
    max_request_tokens: Optional[int] = None


_DREX_2_LIMITS = DrexLimits(
    max_questions=512, max_state_tokens=131_072, max_row_tokens=262_144
)
_DREX_1_5_LIMITS = DrexLimits(
    max_questions=512, max_state_tokens=131_072, max_row_tokens=139_264
)
_DREX_MAX_LIMITS = DrexLimits(
    max_questions=512, max_state_tokens=262_144, max_row_tokens=262_144
)
_DREX_MAX_1_1_LIMITS = DrexLimits(
    max_questions=512,
    max_state_tokens=262_144,
    max_row_tokens=262_144,
    max_request_tokens=262_144,
)


def _drex(input_price_per_million: float, images: bool):
    def data() -> DeepEvalModelData:
        return DeepEvalModelData(
            input_price=input_price_per_million / 1e6,
            output_price=0.0,
            supports_multimodal=images,
        )

    return data


# Retired ids are served by a newer model at that model's price, so they are
# listed under it: `drex-v1.0` is answered by Drex Max, `drex-v1.1` by 1.5.
NACE_MODELS_DATA = ModelDataRegistry(
    {
        "drex-v2.0": _drex(0.06, images=False),
        "drex-latest": _drex(0.06, images=False),
        "drex-v1.5": _drex(0.05, images=False),
        "drex-v1.1": _drex(0.05, images=False),
        "drex-max": _drex(0.03, images=True),
        "drex-max-v1.1": _drex(0.03, images=True),
        "drex-v1.0": _drex(0.03, images=True),
    }
)

NACE_MODEL_LIMITS: Dict[str, DrexLimits] = {
    "drex-v2.0": _DREX_2_LIMITS,
    "drex-latest": _DREX_2_LIMITS,
    "drex-v1.5": _DREX_1_5_LIMITS,
    "drex-v1.1": _DREX_1_5_LIMITS,
    "drex-max": _DREX_MAX_LIMITS,
    "drex-max-v1.1": _DREX_MAX_1_1_LIMITS,
    "drex-v1.0": _DREX_MAX_LIMITS,
}


def drex_limits(model: str) -> DrexLimits:
    return NACE_MODEL_LIMITS.get(model, _DREX_2_LIMITS)
