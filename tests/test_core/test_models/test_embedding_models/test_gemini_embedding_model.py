from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models.embedding_models.gemini_embedding_model import (
    GeminiEmbeddingModel,
)

MODULE = "deepeval.models.embedding_models.gemini_embedding_model"


def _fake_genai(dim=4, n=1, async_mode=False):
    """Fake google.genai module whose Client().models.embed_content returns a
    response with .embeddings[i].values of length `dim`."""
    fake = MagicMock()
    resp = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1] * dim) for _ in range(n)]
    )
    client = fake.Client.return_value
    if async_mode:
        client.aio.models.embed_content = AsyncMock(return_value=resp)
    else:
        client.models.embed_content.return_value = resp
    # EmbedContentConfig is a MagicMock so tests can assert the config object
    # passed to embed_content is exactly what EmbedContentConfig(...) returned.
    fake.types = SimpleNamespace(EmbedContentConfig=MagicMock())
    return fake


@patch(f"{MODULE}.require_dependency")
def test_gemini_embed_text_returns_list_of_float(mock_req, settings):
    fake = _fake_genai(dim=4, n=1)
    mock_req.return_value = fake
    model = GeminiEmbeddingModel(model="gemini-embedding-001", api_key="k")
    vec = model.embed_text("hello")
    assert isinstance(vec, list) and len(vec) == 4
    assert all(isinstance(x, float) for x in vec)
    # Assert the Google GenAI call contract, not just the returned shape.
    fake.Client.return_value.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001",
        contents="hello",
        config=fake.types.EmbedContentConfig.return_value,
    )


@patch(f"{MODULE}.require_dependency")
def test_gemini_embed_texts_returns_list_of_lists(mock_req, settings):
    fake = _fake_genai(dim=4, n=3)
    mock_req.return_value = fake
    model = GeminiEmbeddingModel(model="gemini-embedding-001", api_key="k")
    mat = model.embed_texts(["a", "b", "c"])
    assert isinstance(mat, list) and len(mat) == 3
    assert all(isinstance(v, list) and len(v) == 4 for v in mat)
    assert all(isinstance(x, float) for row in mat for x in row)
    fake.Client.return_value.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001",
        contents=["a", "b", "c"],
        config=fake.types.EmbedContentConfig.return_value,
    )


@patch(f"{MODULE}.require_dependency")
async def test_gemini_a_embed_text_returns_list_of_float(mock_req, settings):
    fake = _fake_genai(dim=4, n=1, async_mode=True)
    mock_req.return_value = fake
    model = GeminiEmbeddingModel(model="gemini-embedding-001", api_key="k")
    vec = await model.a_embed_text("hello")
    assert isinstance(vec, list) and len(vec) == 4
    assert all(isinstance(x, float) for x in vec)
    # Async path must go through client.aio.models.embed_content.
    fake.Client.return_value.aio.models.embed_content.assert_awaited_once_with(
        model="gemini-embedding-001",
        contents="hello",
        config=fake.types.EmbedContentConfig.return_value,
    )


@patch(f"{MODULE}.require_dependency")
async def test_gemini_a_embed_texts_returns_list_of_lists(mock_req, settings):
    fake = _fake_genai(dim=4, n=2, async_mode=True)
    mock_req.return_value = fake
    model = GeminiEmbeddingModel(model="gemini-embedding-001", api_key="k")
    mat = await model.a_embed_texts(["a", "b"])
    assert isinstance(mat, list) and len(mat) == 2
    assert all(isinstance(v, list) and len(v) == 4 for v in mat)
    fake.Client.return_value.aio.models.embed_content.assert_awaited_once_with(
        model="gemini-embedding-001",
        contents=["a", "b"],
        config=fake.types.EmbedContentConfig.return_value,
    )


@patch(f"{MODULE}.require_dependency")
def test_gemini_embedding_get_model_name(mock_req, settings):
    mock_req.return_value = _fake_genai()
    model = GeminiEmbeddingModel(model="gemini-embedding-001", api_key="k")
    assert model.get_model_name() == "gemini-embedding-001 (Gemini)"


@patch(f"{MODULE}.require_dependency")
def test_gemini_embedding_default_model_name(mock_req, settings):
    mock_req.return_value = _fake_genai()
    model = GeminiEmbeddingModel(api_key="k")
    assert model.name == "gemini-embedding-001"


# NOTE: DeepEvalBaseEmbeddingModel.__init__ calls self.load_model() (which
# builds the client) at construction time, so the Vertex AI branch of
# _build_client runs inside the constructor.


def _require_dependency_side_effect(genai_fake, service_account_fake):
    """Route require_dependency to distinct fakes per module so the Vertex
    branch must resolve `Credentials` on the service_account submodule (not on
    the shared genai mock)."""

    def _side_effect(module_name, *args, **kwargs):
        if module_name == "google.oauth2.service_account":
            return service_account_fake
        return genai_fake

    return _side_effect


@patch(f"{MODULE}.require_dependency")
def test_gemini_vertexai_builds_client_with_credentials(mock_req, settings):
    fake = _fake_genai(dim=4, n=1)
    service_account_fake = MagicMock()
    mock_req.side_effect = _require_dependency_side_effect(
        fake, service_account_fake
    )
    key = {"type": "service_account", "project_id": "p"}
    model = GeminiEmbeddingModel(
        model="gemini-embedding-001",
        project="my-project",
        location="us-central1",
        service_account_key=key,
    )
    assert model.should_use_vertexai() is True

    from_info = service_account_fake.Credentials.from_service_account_info
    from_info.assert_called_once_with(
        key,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    fake.Client.assert_called_once_with(
        vertexai=True,
        project="my-project",
        location="us-central1",
        credentials=from_info.return_value,
    )


@patch(f"{MODULE}.require_dependency")
def test_gemini_vertexai_no_service_account_uses_adc(mock_req, settings):
    fake = _fake_genai(dim=4, n=1)
    service_account_fake = MagicMock()
    mock_req.side_effect = _require_dependency_side_effect(
        fake, service_account_fake
    )
    GeminiEmbeddingModel(
        model="gemini-embedding-001",
        project="my-project",
        location="us-central1",
        use_vertexai=True,
    )

    service_account_fake.Credentials.from_service_account_info.assert_not_called()
    fake.Client.assert_called_once_with(
        vertexai=True,
        project="my-project",
        location="us-central1",
        credentials=None,
    )


@patch(f"{MODULE}.require_dependency")
def test_gemini_vertexai_missing_project_location_raises(mock_req, settings):
    mock_req.return_value = _fake_genai()
    with pytest.raises(DeepEvalError, match="project and location are"):
        GeminiEmbeddingModel(model="gemini-embedding-001", use_vertexai=True)


@patch(f"{MODULE}.require_dependency")
def test_gemini_vertexai_invalid_json_service_account_raises(
    mock_req, settings
):
    mock_req.return_value = _fake_genai()
    with pytest.raises(DeepEvalError, match="valid JSON"):
        GeminiEmbeddingModel(
            model="gemini-embedding-001",
            project="my-project",
            location="us-central1",
            service_account_key="not-json",
        )


@patch(f"{MODULE}.require_dependency")
def test_gemini_vertexai_non_object_service_account_raises(mock_req, settings):
    mock_req.return_value = _fake_genai()
    with pytest.raises(DeepEvalError, match="JSON object"):
        GeminiEmbeddingModel(
            model="gemini-embedding-001",
            project="my-project",
            location="us-central1",
            service_account_key="[1, 2, 3]",
        )
