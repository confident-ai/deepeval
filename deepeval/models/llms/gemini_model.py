from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel

from deepeval.key_handler import KEY_FILE_HANDLER, ModelKeyValues
from deepeval.models.base_model import DeepEvalBaseLLM

default_gemini_model = "gemini-1.5-pro"


class GeminiModel(DeepEvalBaseLLM):
    """Class that implements Google Gemini models for text-based evaluation.

    This class provides integration with Google's Gemini models through the Google GenAI SDK,
    supporting text-only inputs for evaluation tasks.
    To use Gemini API, set api_key attribute only.
    To use Vertex AI API, set project and location attributes.

    Attributes:
        model_name: Name of the Gemini model to use
        api_key: Google API key for authentication
        project: Google Cloud project ID
        location: Google Cloud location

    Example:
        ```python
        from deepeval.models import GeminiModel

        # Initialize the model
        model = GeminiModel(
            model_name="gemini-1.5-pro-001",
            api_key="your-api-key"
        )

        # Generate text
        response = model.generate("What is the capital of France?")
        ```
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = (
            model_name
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GEMINI_MODEL_NAME)
            or default_gemini_model
        )

        # Get API key from key handler if not provided
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_API_KEY
        )
        self.project = project or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_CLOUD_PROJECT
        )
        self.location = location or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_CLOUD_LOCATION
        )
        self.use_vertexai = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name, **kwargs)

    def should_use_vertexai(self):
        """Checks if the model should use Vertex AI for generation.

        This is determined first by the value of `GOOGLE_GENAI_USE_VERTEXAI`
        environment variable. If not set, it checks for the presence of the
        project and location.

        Returns:
            True if the model should use Vertex AI, False otherwise
        """
        if self.use_vertexai is not None:
            return self.use_vertexai.lower() == "yes"
        if self.project and self.location:
            return True
        else:
            return False

    def load_model(self, *args, **kwargs):
        """Creates a client.
        With Gen AI SDK, model is set at inference time, so there is no
        model to load and initialize.
        This method name is kept for compatibility with other LLMs.

        Returns:
            A GenerativeModel instance configured for evaluation.
        """
        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise ValueError(
                    "When using Vertex AI API, both project and location are required."
                    "Either provide them as arguments or set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables, "
                    "or set them in your DeepEval configuration."
                )

            # Create client for Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                **self.kwargs,
            )
        else:
            if not self.api_key:
                raise ValueError(
                    "Google API key is required. Either provide it directly, set GOOGLE_API_KEY environment variable, "
                    "or set it in your DeepEval configuration."
                )
            # Create client for Gemini API
            self.client = genai.Client(api_key=self.api_key, **self.kwargs)

        # Configure default model generation settings
        self.model_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        return self.client.models

    ###############################################
    # Generate functions
    ###############################################

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        if schema is not None:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.parsed, 0
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> str:
        """Asynchronously generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        if schema is not None:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.parsed, 0
        else:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    ###############################################
    # Other generate functions
    ###############################################

    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        """Generates a raw response with top log probabilities.

        Mirrors the OpenAI interface: returns (raw_response, cost).
        Cost is 0.0 as Gemini usage pricing calculation is not implemented here.
        """
        # Enable logprobs in generation config when supported
        # Gemini only supports logprobs in the range [0, 20)
        safe_logprobs = max(0, min(int(top_logprobs), 19))

        raw = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=self.model_safety_settings,
                temperature=self.temperature,
                # Gemini logprobs configuration
                response_logprobs=True,
                logprobs=safe_logprobs,
            ),
        )
        wrapped = GeminiModel.transform_gemini_to_openai_like(raw)
        return wrapped, 0.0

    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        """Async version: returns raw response with top log probabilities.

        Returns (raw_response, cost). Cost is 0.0 for now.
        """
        # Gemini only supports logprobs in the range [0, 20)
        safe_logprobs = max(0, min(int(top_logprobs), 19))

        raw = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=self.model_safety_settings,
                temperature=self.temperature,
                response_logprobs=True,
                logprobs=safe_logprobs,
            ),
        )
        wrapped = GeminiModel.transform_gemini_to_openai_like(raw)
        return wrapped, 0.0

    ###############################################
    # Utilities
    ###############################################

    @staticmethod
    def transform_gemini_to_openai_like(raw: Any) -> Any:
        """Transform a Gemini GenerateContentResponse into an OpenAI-like ChatCompletion.

        Aligns with the actual Gemini response structure (snake_case):
        - choices[0].message.content is built by concatenating candidate.content.parts[*].text
        - choices[0].logprobs.content is built from candidate.logprobs_result:
            - chosen_candidates: the emitted tokens per position
            - top_candidates[i].candidates: top alternatives for position i with log_probability
        """
        # Extract primary candidate
        candidates = getattr(raw, "candidates", None) or []
        candidate0 = candidates[0] if candidates else None

        # Build text content (prefer raw.text if available)
        text_content = getattr(raw, "text", None) or ""
        if (
            not text_content
            and candidate0
            and getattr(candidate0, "content", None)
        ):
            parts = getattr(candidate0.content, "parts", None) or []
            texts = []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
            text_content = "".join(texts)

        # Structures to mimic OpenAI
        class _TokenTopLogprob:
            def __init__(self, token: str, logprob: float):
                self.token = token
                self.logprob = logprob

        class _TokenLogprob:
            def __init__(
                self, token: str, top_logprobs: list[_TokenTopLogprob]
            ):
                self.token = token
                self.top_logprobs = top_logprobs

        class _Logprobs:
            def __init__(self, content: list[_TokenLogprob]):
                self.content = content

        class _Message:
            def __init__(self, content: str):
                self.content = content

        class _Choice:
            def __init__(self, message: _Message, logprobs: _Logprobs):
                self.message = message
                self.logprobs = logprobs

        class _Completion:
            def __init__(self, choices: list[_Choice]):
                self.choices = choices

        # Convert logprobs using the explicit Gemini fields
        converted_tokens: list[_TokenLogprob] = []
        try:
            logprobs_result = (
                getattr(candidate0, "logprobs_result", None)
                if candidate0
                else None
            )
            if logprobs_result is not None:
                chosen = (
                    getattr(logprobs_result, "chosen_candidates", None) or []
                )
                top = getattr(logprobs_result, "top_candidates", None) or []
                length = min(len(chosen), len(top))
                for i in range(length):
                    chosen_token = getattr(chosen[i], "token", "") or ""
                    per_token_top = []
                    top_i = getattr(top[i], "candidates", None) or []
                    for cand in top_i:
                        tok = getattr(cand, "token", "") or ""
                        lp = getattr(cand, "log_probability", None)
                        if tok != "" and isinstance(lp, (int, float)):
                            per_token_top.append(
                                _TokenTopLogprob(tok, float(lp))
                            )
                    converted_tokens.append(
                        _TokenLogprob(chosen_token, per_token_top)
                    )
        except Exception:
            converted_tokens = []

        logprobs_obj = _Logprobs(converted_tokens)
        wrapped = _Completion([_Choice(_Message(text_content), logprobs_obj)])
        return wrapped

    ###############################################
    # Model
    ###############################################

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name
