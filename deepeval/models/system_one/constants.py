from deepeval.models.base_model import DeepEvalModelData
from deepeval.models.llms.constants import ModelDataRegistry

DEFAULT_TYPESAFE_MODEL = "jev-latest"

# Jev is priced on input tokens only; output tokens are free.
_JEV_INPUT_PRICE = 0.042 / 1e6


def _jev() -> DeepEvalModelData:
    return DeepEvalModelData(input_price=_JEV_INPUT_PRICE, output_price=0.0)


TYPESAFE_MODELS_DATA = ModelDataRegistry(
    {
        "jev-1.13.0": _jev,
        "jev-latest": _jev,
        "jev-preview": _jev,
    }
)
