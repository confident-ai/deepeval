import base64
import json
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import SecretStr

from deepeval._version import __version__
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.system_one.constants import (
    DEFAULT_NACE_MODEL,
    DREX_IMAGE_MIME_TYPES,
    DREX_MAX_BODY_BYTES,
    DREX_MAX_IMAGE_BYTES,
    DREX_MAX_IMAGES,
    DREX_MAX_MODEL,
    DREX_MAX_TOTAL_IMAGE_BYTES,
    NACE_BASE_URL,
    NACE_MODELS_DATA,
    OPENROUTER_NACE_PREFIX,
    drex_limits,
)
from deepeval.models.system_one.limits import (
    SystemOneContextLimitError,
    check_drex_budget,
    estimate_tokens,
)
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
    SystemOneQuestion,
)
from deepeval.models.utils import EvaluationCost, require_secret_api_key
from deepeval.utils import require_dependency

retry_nace = create_retry_decorator(PS.NACE)

# Sent as `X-Nace-Client` so Drex can attribute requests to deepeval.
CLIENT_INFO = f"deepeval/{__version__}"
NACE_BILLING_URL = "https://console.nace.ai/dashboard/billing"
# JSON punctuation and headers around the parts the size check measures.
_BODY_MARGIN_BYTES = 64 * 1024


def is_openrouter(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


@contextmanager
def _drex_errors(state: Any):
    """Turn Drex refusals into errors that say what to change: a token or
    body-size limit (422 "Request too long", or the hosting layer's 413)
    becomes ``SystemOneContextLimitError`` (so ``hybrid`` hands the decision
    to the LLM), any other 422 lists each issue's path and message, and a
    402 points at Billing. Everything else, retryable ones included, passes
    through unchanged."""
    try:
        yield
    except Exception as error:
        status = getattr(error, "status", None)
        issues = getattr(error, "issues", None) or []
        message = getattr(error, "message", None) or str(error)
        request_id = getattr(error, "request_id", None)
        suffix = f" (request_id={request_id})" if request_id else ""
        if status == 413 or (
            status == 422 and message.startswith("Request too long")
        ):
            raise SystemOneContextLimitError(
                f"Drex rejected the request as too large: {message}{suffix}",
                estimated_tokens=estimate_tokens(state),
                limit_tokens=None,
            ) from error
        if status == 422:
            details = "; ".join(
                f"{issue.get('path', '')}: {issue.get('message', '')}"
                for issue in issues
            )
            raise DeepEvalError(
                f"Drex rejected the request (422 invalid_request_error): "
                f"{details or message}{suffix}"
            ) from error
        if status == 402:
            raise DeepEvalError(
                f"Nace has no available funds for this API key (402): "
                f"{message} Add funds at {NACE_BILLING_URL}.{suffix}"
            ) from error
        raise


def _text(value: Any) -> Optional[str]:
    """Drex reads instructions and criteria as text only."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


class NaceModel(DeepEvalBaseSystemOneModel):
    """Nace's Drex decision models as a System One model.

    ``model`` is any id ``GET /v1/models`` lists (``drex-max-v1.1`` by
    default, which reads images too); a text-only model refuses images
    instead of dropping them. ``base_url``
    defaults to Nace; pointing it at OpenRouter (``https://openrouter.ai/api``)
    or another TypeSafe-compatible host runs the same code there."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        **kwargs,
    ):
        settings = get_settings()

        base_url = base_url or settings.NACE_BASE_URL
        self.base_url = str(base_url or NACE_BASE_URL).rstrip("/")
        self.openrouter = is_openrouter(self.base_url)

        # The Nace key never goes to OpenRouter.
        if api_key is not None:
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        elif self.openrouter:
            self.api_key = settings.OPENROUTER_API_KEY
        else:
            self.api_key = settings.NACE_API_KEY

        model = model or settings.NACE_MODEL_NAME or DEFAULT_NACE_MODEL
        # OpenRouter's `nace-ai/` ids name the same models; `_body` adds the
        # prefix back for OpenRouter.
        model = model.removeprefix(OPENROUTER_NACE_PREFIX)

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.NACE_COST_PER_INPUT_TOKEN
        )
        if cost_per_input_token is not None and cost_per_input_token < 0:
            raise DeepEvalError("NACE_COST_PER_INPUT_TOKEN must be >= 0.")
        self.cost_per_input_token = cost_per_input_token

        self.kwargs = kwargs
        self._client = None
        self._async_client = None
        super().__init__(model)

    ###############################################
    # Decide
    ###############################################

    @retry_nace
    def decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
        images: Optional[List[Any]] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        model = self._model_for(images)
        self.check_budget(state, questions, images)
        client = self.load_model()
        uploads: List[Any] = []
        try:
            # Uploads too: a 402 or 413 there reads like one from the call.
            with _drex_errors(state):
                request = self._request(
                    state, questions, model, images, uploads, client
                )
                response = client.system_one(**request)
        finally:
            for uploaded in uploads:
                self._delete_upload(client, uploaded)
        return self._from_sdk_response(response)

    @retry_nace
    async def a_decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
        images: Optional[List[Any]] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        model = self._model_for(images)
        self.check_budget(state, questions, images)
        client = self.load_model(async_mode=True)
        uploads: List[Any] = []
        try:
            with _drex_errors(state):
                request = await self._a_request(
                    state, questions, model, images, uploads, client
                )
                response = await client.system_one(**request)
        finally:
            for uploaded in uploads:
                await self._a_delete_upload(client, uploaded)
        return self._from_sdk_response(response)

    def supports_images(self) -> bool:
        return _reads_images(self.name)

    def check_budget(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
        images: Optional[List[Any]] = None,
    ) -> None:
        model = self._model_for(images)
        if not questions:
            raise DeepEvalError(
                "NaceModel.decide requires at least one question."
            )
        check_drex_budget(
            state,
            questions,
            model=model,
            limits=drex_limits(model),
            image_count=len(images or []),
        )
        body = len(_dumps({"state": state, "questions": questions}))
        if body + _BODY_MARGIN_BYTES > DREX_MAX_BODY_BYTES:
            raise SystemOneContextLimitError(
                f"System One request body ({body} bytes before images) "
                f"exceeds Drex's {DREX_MAX_BODY_BYTES}-byte limit. Send less "
                f"state or judge this test case with the LLM.",
                estimated_tokens=estimate_tokens(state),
                limit_tokens=None,
            )

    def _model_for(self, images: Optional[List[Any]]) -> str:
        if not images or _reads_images(self.name):
            return self.name
        raise DeepEvalError(
            f"{self.name} reads text only, and this request has "
            f"{len(images)} image(s). Use a Drex Max model such as "
            f"NaceModel(model='{DEFAULT_NACE_MODEL}'), the default."
        )

    ###############################################
    # Translation
    ###############################################

    def _request(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
        model: str,
        images: Optional[List[Any]],
        uploads: List[Any],
        client: Any,
    ) -> Dict[str, Any]:
        sdk_questions = self._to_sdk_questions(questions)
        wire_images: List[Any] = []
        if images:
            payloads = _image_payloads(images)
            budget = self._inline_budget(state, sdk_questions)
            for data, url, upload in self._plan_images(payloads, budget):
                if upload:
                    uploaded = client.documents.upload(
                        data, file_name=_upload_name(url)
                    )
                    uploads.append(uploaded)
                    wire_images.append(_workspace_file(uploaded))
                else:
                    wire_images.append(url)
        return self._body(state, sdk_questions, model, wire_images)

    async def _a_request(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
        model: str,
        images: Optional[List[Any]],
        uploads: List[Any],
        client: Any,
    ) -> Dict[str, Any]:
        sdk_questions = self._to_sdk_questions(questions)
        wire_images: List[Any] = []
        if images:
            payloads = _image_payloads(images)
            budget = self._inline_budget(state, sdk_questions)
            for data, url, upload in self._plan_images(payloads, budget):
                if upload:
                    uploaded = await client.documents.upload(
                        data, file_name=_upload_name(url)
                    )
                    uploads.append(uploaded)
                    wire_images.append(_workspace_file(uploaded))
                else:
                    wire_images.append(url)
        return self._body(state, sdk_questions, model, wire_images)

    def _inline_budget(self, state: Any, sdk_questions: Dict[str, Any]) -> int:
        used = len(_dumps({"state": state, "questions": sdk_questions}))
        return DREX_MAX_BODY_BYTES - _BODY_MARGIN_BYTES - used

    def _plan_images(
        self, payloads: List[Tuple[bytes, str]], budget: int
    ) -> List[Tuple[bytes, str, bool]]:
        """Inline every image that fits in the 4 MiB body, uploading the
        largest first until the rest do. OpenRouter takes inline images only."""
        upload = [False] * len(payloads)
        inline = sum(len(url) for _, url in payloads)
        for i in sorted(
            range(len(payloads)), key=lambda i: -len(payloads[i][1])
        ):
            if inline <= budget:
                break
            if self.openrouter:
                raise DeepEvalError(
                    f"The images in this request add up to {inline} bytes "
                    f"inline, over the {DREX_MAX_BODY_BYTES}-byte request "
                    f"body OpenRouter forwards to Drex. Downscale them (Drex "
                    f"Max reads at most 1280x1280 pixels anyway) or use the "
                    f"Nace base URL, which uploads large images."
                )
            upload[i] = True
            inline -= len(payloads[i][1])
        return [(data, url, up) for (data, url), up in zip(payloads, upload)]

    def _body(
        self,
        state: Any,
        sdk_questions: Dict[str, Any],
        model: str,
        images: List[Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "state": state,
            "questions": sdk_questions,
            "model": model,
        }
        if not self.openrouter:
            if images:
                body["images"] = images
            return body
        # OpenRouter names the model under its `nace-ai/` namespace, needs
        # `instructions` on every question and rejects a top-level `images`:
        # images go in a content array that replaces the state.
        body["model"] = OPENROUTER_NACE_PREFIX + model
        for question in sdk_questions.values():
            question.setdefault("instructions", "")
        if images:
            text = state if isinstance(state, str) else _dumps(state).decode()
            body["state"] = [{"type": "text", "text": text}] + [
                {"type": "image_url", "image_url": {"url": url}}
                for url in images
            ]
        return body

    def _to_sdk_questions(
        self, questions: Dict[str, SystemOneQuestion]
    ) -> Dict[str, Any]:
        if not questions:
            raise DeepEvalError(
                "NaceModel.decide requires at least one question."
            )
        sdk: Dict[str, Any] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                entry: Dict[str, Any] = {"type": "noul"}
                if question.true is not None or question.false is not None:
                    entry["criteria"] = {
                        "true": _text(question.true),
                        "false": _text(question.false),
                    }
            elif isinstance(question, ChoiceQuestion):
                entry = {
                    "type": "choice",
                    "criteria": {
                        str(label): _text(description)
                        for label, description in question.options.items()
                    },
                }
            elif isinstance(question, ScoreQuestion):
                entry = {
                    "type": "score",
                    "criteria": [_text(level) for level in question.levels],
                }
            else:
                raise DeepEvalError(
                    f"Unsupported System One question type: {type(question)}"
                )
            # Drex accepts instructions as text or no key at all, never null.
            instructions = _text(question.instructions)
            if instructions:
                entry["instructions"] = instructions
            sdk[key] = entry
        return sdk

    def _from_sdk_response(
        self, response: Any
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        answers = SystemOneAnswers()
        for key, answer in response.nouls.items():
            answers.nouls[key] = NoulAnswer(probability=answer.noul)
        for key, answer in response.choices.items():
            answers.choices[key] = ChoiceAnswer(
                choice=answer.choice,
                probabilities=dict(answer.probabilities),
                confidence=answer.confidence,
            )
        for key, answer in response.scores.items():
            answers.scores[key] = ScoreAnswer(
                score=answer.score,
                probabilities={
                    int(level): p for level, p in answer.probabilities.items()
                },
                confidence=answer.confidence,
            )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or 0
        return answers, self.calculate_cost(input_tokens, output_tokens)

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        # Drex bills input tokens only; image tokens are among them.
        price = self.cost_per_input_token
        if price is None:
            price = NACE_MODELS_DATA.get(self.name).input_price
        if price is None:
            return None
        return EvaluationCost(input_tokens * price, input_tokens, output_tokens)

    def _delete_upload(self, client: Any, uploaded: Any) -> None:
        try:
            client.documents.delete_file(
                uploaded.workspace_id, uploaded.file_id
            )
        except Exception:  # noqa: BLE001 - best effort; the answer stands
            pass

    async def _a_delete_upload(self, client: Any, uploaded: Any) -> None:
        try:
            await client.documents.delete_file(
                uploaded.workspace_id, uploaded.file_id
            )
        except Exception:  # noqa: BLE001 - best effort; the answer stands
            pass

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        module = require_dependency(
            "nace_sdk",
            provider_label="NaceModel",
            install_hint="Install it with `pip install nace-sdk` "
            "(Python 3.11+).",
        )
        if not async_mode:
            if self._client is None:
                self._client = self._build_client(module, module.NaceClient)
            return self._client
        if self._async_client is None:
            self._async_client = self._build_client(
                module, module.AsyncNaceClient
            )
        return self._async_client

    def _build_client(self, module, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenRouter" if self.openrouter else "Nace",
            env_var_name=(
                "OPENROUTER_API_KEY" if self.openrouter else "NACE_API_KEY"
            ),
            param_hint="`api_key` to NaceModel(...)",
        )
        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            model=self.name,
            client_info=CLIENT_INFO,
            **self.kwargs,
        )
        if self.openrouter:
            kw["headers"] = {
                "HTTP-Referer": "https://nace.ai",
                "X-OpenRouter-Title": "Drex for deepeval",
                **(kw.get("headers") or {}),
            }
        if not sdk_retries_for(PS.NACE) and "retry" not in kw:
            kw["retry"] = module.RetryPolicy(max_retries=0)
        return cls(**kw)

    def get_model_name(self):
        return f"{self.name} (Nace)"


def _reads_images(model: str) -> bool:
    return bool(
        NACE_MODELS_DATA.get(model).supports_multimodal
        or model.startswith(DREX_MAX_MODEL)
    )


def _dumps(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda v: (
            v.model_dump() if hasattr(v, "model_dump") else str(v)
        ),
    ).encode()


def _image_payloads(images: List[Any]) -> List[Tuple[bytes, str]]:
    """Each image's bytes and data URL, checked against Drex Max's image
    limits so a bad image fails here, naming the image, not as a 422."""
    from deepeval.models.llms.multimodal import decode_image, fetch_remote_image
    from deepeval.test_case import MLLMImage

    if len(images) > DREX_MAX_IMAGES:
        raise DeepEvalError(
            f"Drex Max reads at most {DREX_MAX_IMAGES} images per request; "
            f"this one has {len(images)}."
        )
    payloads: List[Tuple[bytes, str]] = []
    for position, image in enumerate(images, start=1):
        if not isinstance(image, MLLMImage):
            raise DeepEvalError(
                f"Image {position} is a {type(image).__name__}; NaceModel "
                f"takes MLLMImage."
            )
        if image.url and not image.local and not image.dataBase64:
            # Drex never fetches a URL, so the bytes are sent instead.
            data, _ = fetch_remote_image(image)
        else:
            data = decode_image(image, DeepEvalError)
        mime = _sniff_mime(data)
        if mime not in DREX_IMAGE_MIME_TYPES:
            raise DeepEvalError(
                f"Image {position} ({image.filename or image.url or 'inline'}"
                f") is not PNG, JPEG or WebP, the formats Drex Max reads."
            )
        if len(data) > DREX_MAX_IMAGE_BYTES:
            raise DeepEvalError(
                f"Image {position} is {len(data)} bytes, over Drex Max's "
                f"{DREX_MAX_IMAGE_BYTES}-byte limit per image."
            )
        encoded = base64.b64encode(data).decode("ascii")
        payloads.append((data, f"data:{mime};base64,{encoded}"))
    total = sum(len(data) for data, _ in payloads)
    if total > DREX_MAX_TOTAL_IMAGE_BYTES:
        raise DeepEvalError(
            f"The images add up to {total} bytes, over Drex Max's "
            f"{DREX_MAX_TOTAL_IMAGE_BYTES}-byte limit per request."
        )
    return payloads


def _sniff_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _workspace_file(uploaded: Any) -> Dict[str, str]:
    return {
        "type": "workspace_file",
        "workspace_id": uploaded.workspace_id,
        "file_id": uploaded.file_id,
    }


def _upload_name(data_url: str) -> str:
    extension = data_url[len("data:image/") : data_url.index(";")]
    return f"deepeval-{uuid.uuid4().hex}.{extension}"
