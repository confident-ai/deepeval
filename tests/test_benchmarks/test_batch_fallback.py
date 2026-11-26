from deepeval.benchmarks.utils import should_use_batch
from deepeval.models import DeepEvalBaseLLM


class _TextOnlyModel(DeepEvalBaseLLM):
    """A custom model that does not implement batch_generate()."""

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        return "42"

    async def a_generate(self, *args, **kwargs):
        return "42"

    def get_model_name(self, *args, **kwargs):
        return "text-only"


class _BatchModel(_TextOnlyModel):
    """A custom model that does implement batch_generate()."""

    def batch_generate(self, prompts, schemas=None):
        return ["42"] * len(prompts)


def _model(cls):
    # Bypass __init__ (which calls load_model); should_use_batch only needs
    # the class/instance method resolution.
    return cls.__new__(cls)


def test_no_batch_size_means_no_batching():
    assert should_use_batch(_model(_TextOnlyModel)) is False


def test_inherited_not_implemented_batch_generate_falls_back():
    # DeepEvalBaseLLM.batch_generate only raises NotImplementedError. Merely
    # inheriting it must not enable batching.
    assert should_use_batch(_model(_TextOnlyModel), batch_size=2) is False


def test_overridden_batch_generate_enables_batching():
    assert should_use_batch(_model(_BatchModel), batch_size=2) is True


def test_duck_typed_model_with_batch_generate_enables_batching():
    class _DuckTypedModel:
        def batch_generate(self, prompts, schemas=None):
            return ["42"] * len(prompts)

    assert should_use_batch(_DuckTypedModel(), batch_size=2) is True
