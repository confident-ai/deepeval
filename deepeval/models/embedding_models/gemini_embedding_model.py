import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.utils import require_secret_api_key
from deepeval.utils import require_dependency
from deepeval.constants import ProviderSlug as PS

if TYPE_CHECKING:
    from google.genai import Client

default_gemini_embedding_model = "gemini-embedding-001"

retry_gemini = create_retry_decorator(PS.GOOGLE)


class GeminiEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """Google Gemini embedding model (Gemini API or Vertex AI).

    Uses the Google GenAI SDK (`google-genai`). Set `api_key` to use the Gemini
    API, or set `project`/`location` (and optionally `service_account_key`) to
    use Vertex AI.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_key: Optional[Union[str, Dict[str, str]]] = None,
        use_vertexai: Optional[bool] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = (
            model
            or settings.GEMINI_EMBEDDING_MODEL_NAME
            or default_gemini_embedding_model
        )

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.GOOGLE_API_KEY

        self.project = project or settings.GOOGLE_CLOUD_PROJECT
        location = (
            location if location is not None else settings.GOOGLE_CLOUD_LOCATION
        )
        self.location = str(location).strip() if location is not None else None
        self.use_vertexai = (
            use_vertexai
            if use_vertexai is not None
            else settings.GOOGLE_GENAI_USE_VERTEXAI
        )

        self.service_account_key: Optional[SecretStr] = None
        if service_account_key is None:
            self.service_account_key = settings.GOOGLE_SERVICE_ACCOUNT_KEY
        elif isinstance(service_account_key, dict):
            self.service_account_key = SecretStr(
                json.dumps(service_account_key)
            )
        else:
            str_value = str(service_account_key).strip()
            self.service_account_key = (
                SecretStr(str_value) if str_value else None
            )

        # Raw kwargs destined for the underlying Client constructor.
        self.kwargs = kwargs
        self.generation_kwargs = dict(generation_kwargs or {})

        self._module = self._require_module()
        super().__init__(model)

    ###############################################
    # Embedding functions
    ###############################################

    @retry_gemini
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model()
        response = client.models.embed_content(
            model=self.name,
            contents=text,
            config=self._embed_config(),
        )
        return list(response.embeddings[0].values)

    @retry_gemini
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model()
        response = client.models.embed_content(
            model=self.name,
            contents=texts,
            config=self._embed_config(),
        )
        return [list(e.values) for e in response.embeddings]

    @retry_gemini
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model()
        response = await client.aio.models.embed_content(
            model=self.name,
            contents=text,
            config=self._embed_config(),
        )
        return list(response.embeddings[0].values)

    @retry_gemini
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model()
        response = await client.aio.models.embed_content(
            model=self.name,
            contents=texts,
            config=self._embed_config(),
        )
        return [list(e.values) for e in response.embeddings]

    ###############################################
    # Model
    ###############################################

    def should_use_vertexai(self) -> bool:
        if self.use_vertexai is not None:
            return self.use_vertexai
        if self.project and self.location:
            return True
        return False

    def load_model(self):
        """Creates and returns a google-genai Client.

        The Gen AI SDK sets the model at inference time, so there is no model to
        load. A single client serves both sync (`client.models`) and async
        (`client.aio.models`) calls.
        """
        return self._build_client()

    def _require_module(self):
        return require_dependency(
            "google.genai",
            provider_label="GeminiEmbeddingModel",
            install_hint="Install it with `pip install google-genai`.",
        )

    def _require_service_account(self):
        return require_dependency(
            "google.oauth2.service_account",
            provider_label="GeminiEmbeddingModel",
            install_hint="Install it with `pip install google-auth`.",
        )

    def _embed_config(self):
        return self._module.types.EmbedContentConfig(**self.generation_kwargs)

    def _build_client(self) -> "Client":
        client_kwargs = dict(self.kwargs or {})

        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise DeepEvalError(
                    "When using Vertex AI API, both project and location are "
                    "required. Either provide them as arguments or set "
                    "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in your "
                    "DeepEval configuration."
                )

            credentials = None
            if self.service_account_key is not None:
                service_account_key_json = require_secret_api_key(
                    self.service_account_key,
                    provider_label="Google Gemini",
                    env_var_name="GOOGLE_SERVICE_ACCOUNT_KEY",
                    param_hint="`service_account_key` to GeminiEmbeddingModel(...)",
                )
                try:
                    service_account_key = json.loads(service_account_key_json)
                except Exception as e:
                    raise DeepEvalError(
                        "GOOGLE_SERVICE_ACCOUNT_KEY must be valid JSON for a "
                        "Google service account."
                    ) from e
                if not isinstance(service_account_key, dict):
                    raise DeepEvalError(
                        "GOOGLE_SERVICE_ACCOUNT_KEY must decode to a JSON object."
                    )
                service_account = self._require_service_account()
                credentials = (
                    service_account.Credentials.from_service_account_info(
                        service_account_key,
                        scopes=[
                            "https://www.googleapis.com/auth/cloud-platform"
                        ],
                    )
                )

            return self._module.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                credentials=credentials,
                **client_kwargs,
            )

        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Google Gemini",
            env_var_name="GOOGLE_API_KEY",
            param_hint="`api_key` to GeminiEmbeddingModel(...)",
        )
        return self._module.Client(api_key=api_key, **client_kwargs)

    def get_model_name(self):
        return f"{self.name} (Gemini)"
