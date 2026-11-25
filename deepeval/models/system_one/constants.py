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

DEFAULT_OPENAI_DECISION_MODEL = "gpt-6-luna"

# OpenAI Decisions bills uncached input tokens only; output tokens are free.
_GPT_6_LUNA_INPUT_PRICE = 0.10 / 1e6

OPENAI_DECISIONS_MAX_QUESTIONS = 400


def _gpt_6_luna() -> DeepEvalModelData:
    return DeepEvalModelData(
        input_price=_GPT_6_LUNA_INPUT_PRICE, output_price=0.0
    )


OPENAI_DECISION_MODELS_DATA = ModelDataRegistry(
    {
        "gpt-6-luna": _gpt_6_luna,
    }
)
