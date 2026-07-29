"""Regression test for issue #2985.

DeepEvalBaseLLM.load_model (and siblings) was annotated as returning
the abstract base class itself (e.g. -> "DeepEvalBaseLLM"), but the
docstring says "A model object" and every built-in integration returns
a provider client (OpenAI, AzureOpenAI, etc.).  A custom subclass that
honestly annotated its override with the concrete client type triggered
pyright's reportIncompatibleMethodOverride.

The fix: change the abstract return annotation to ``Any`` so that
subclasses are free to return whatever concrete model object they wrap.
"""

import typing
import inspect

import pytest

from deepeval.models.base_model import (
    DeepEvalBaseLLM,
    DeepEvalBaseModel,
    DeepEvalBaseEmbeddingModel,
)


# ---------------------------------------------------------------------------
# 1.  The abstract return annotation must be Any (or at least not the
#     base class itself) so subclasses can return a concrete client.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls",
    [DeepEvalBaseLLM, DeepEvalBaseModel, DeepEvalBaseEmbeddingModel],
    ids=["DeepEvalBaseLLM", "DeepEvalBaseModel", "DeepEvalBaseEmbeddingModel"],
)
def test_load_model_return_annotation_is_not_self_type(cls):
    """load_model must NOT be annotated as returning the base class."""
    sig = inspect.signature(cls.load_model)
    ret = sig.return_annotation
    # The annotation should not be the class itself (string or type)
    assert ret is not cls, (
        f"{cls.__name__}.load_model return annotation must not be the "
        f"base class itself (got {ret!r})"
    )
    # Also reject the stringified form
    if isinstance(ret, str):
        assert ret != cls.__name__, (
            f"{cls.__name__}.load_model return annotation must not be "
            f"the string '{cls.__name__}'"
        )


# ---------------------------------------------------------------------------
# 2.  A concrete subclass that returns a plain object must type-check
#     without forcing the user to suppress pyright rules.
# ---------------------------------------------------------------------------

class _FakeClient:
    """Stand-in for a provider client (OpenAI, Anthropic, …)."""
    def invoke(self, prompt: str) -> str:
        return "ok"


class _MyLLM(DeepEvalBaseLLM):
    def load_model(self):
        return _FakeClient()

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    def get_model_name(self) -> str:
        return "fake"


class _MyEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def load_model(self):
        return _FakeClient()

    def embed_text(self, text: str) -> typing.List[float]:
        return [0.0]

    async def a_embed_text(self, text: str) -> typing.List[float]:
        return [0.0]

    def embed_texts(self, texts: typing.List[str]) -> typing.List[typing.List[float]]:
        return [[0.0]]

    async def a_embed_texts(self, texts: typing.List[str]) -> typing.List[typing.List[float]]:
        return [[0.0]]

    def get_model_name(self) -> str:
        return "fake-embed"


def test_subclass_can_return_arbitrary_client():
    """Subclass load_model returns a non-DeepEvalBaseLLM object at runtime."""
    llm = _MyLLM(model="fake")
    assert isinstance(llm.model, _FakeClient)
    assert llm.generate("hello") == "ok"


def test_embedding_subclass_can_return_arbitrary_client():
    emb = _MyEmbeddingModel(model="fake")
    assert isinstance(emb.model, _FakeClient)
    assert emb.embed_text("hello") == [0.0]


# ---------------------------------------------------------------------------
# 3.  Verify the annotation is Any specifically (strongest guarantee).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls",
    [DeepEvalBaseLLM, DeepEvalBaseModel, DeepEvalBaseEmbeddingModel],
)
def test_load_model_return_annotation_is_any(cls):
    sig = inspect.signature(cls.load_model)
    ret = sig.return_annotation
    assert ret is typing.Any, (
        f"{cls.__name__}.load_model return annotation should be "
        f"typing.Any, got {ret!r}"
    )
