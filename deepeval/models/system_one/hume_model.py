import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import Multipart, SpeechTransport
from deepeval.models.system_one.constants import (
    DEFAULT_HUME_DECISION_MODEL,
    HUME_DECISION_MODELS_DATA,
    HUME_MAX_CHOICES,
    HUME_MAX_QUESTIONS,
)
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    SystemOneAnswers,
)
from deepeval.models.utils import require_secret_api_key

retry_hume = create_retry_decorator(PS.HUME)

HUME_API_URL = "https://api.cloud.hume.ai/v0"
HUME_DECISIONS_PATH = "/decisions/file"
HUME_MAX_INSTRUCTIONS_CHARS = 500
HUME_MAX_CRITERION_CHARS = 300
HUME_MAX_CHOICE_NAME_CHARS = 50
HUME_MAX_CHOICE_DESCRIPTION_CHARS = 100
HUME_MAX_CONTEXT_CHARS = 1_000
HUME_TIMEOUT_SECONDS = 60.0


def _clipped(text: Any, limit: int) -> str:
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class HumeDecisionModel(DeepEvalBaseAudioDecisionModel):
    """Hume's Audio Decisions API: yes/no and pick-one questions answered
    by listening to a WAV or MP3 recording between 1 second and 5 minutes."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        timeout: float = HUME_TIMEOUT_SECONDS,
    ):
        self.api_key: Optional[SecretStr] = (
            SecretStr(api_key)
            if api_key is not None
            else get_settings().HUME_API_KEY
        )
        if cost_per_input_token is not None and cost_per_input_token < 0:
            raise DeepEvalError("cost_per_input_token must be >= 0.")
        model = model or DEFAULT_HUME_DECISION_MODEL
        self.model_data = HUME_DECISION_MODELS_DATA.get(model)
        if cost_per_input_token is not None:
            self.model_data.input_price = cost_per_input_token
        self.timeout = timeout
        super().__init__(model)

    def load_model(self) -> SpeechTransport:
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Hume",
            env_var_name="HUME_API_KEY",
            param_hint="`api_key` to HumeDecisionModel(...)",
        )
        return SpeechTransport(
            provider_label="Hume",
            base_url=HUME_API_URL,
            headers={"X-Hume-Api-Key": api_key},
            timeout=self.timeout,
        )

    def get_model_name(self, *args, **kwargs) -> str:
        return f"Hume {self.name}"

    def calculate_cost(self, input_tokens: int) -> Optional[float]:
        if self.model_data.input_price is None:
            return None
        return input_tokens * self.model_data.input_price

    def decide(
        self,
        audio: bytes,
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        context: Optional[str] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        multipart, names_by_key = _to_multipart(audio, questions, context)
        data = self._post(multipart)
        return self._from_wire(data, questions, names_by_key)

    async def a_decide(
        self,
        audio: bytes,
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        context: Optional[str] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        multipart, names_by_key = _to_multipart(audio, questions, context)
        data = await self._a_post(multipart)
        return self._from_wire(data, questions, names_by_key)

    @retry_hume
    def _post(self, multipart: Multipart) -> Dict[str, Any]:
        return _checked(
            self.model.request_json(
                "POST", HUME_DECISIONS_PATH, multipart=multipart
            )
        )

    @retry_hume
    async def _a_post(self, multipart: Multipart) -> Dict[str, Any]:
        return _checked(
            await self.model.a_request_json(
                "POST", HUME_DECISIONS_PATH, multipart=multipart
            )
        )

    def _from_wire(
        self,
        data: Dict[str, Any],
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        names_by_key: Dict[str, Dict[str, str]],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        answers = SystemOneAnswers()
        for key, question in questions.items():
            answer = data["answers"].get(key)
            if not isinstance(answer, dict):
                continue
            if isinstance(question, NoulQuestion):
                probability = _probability(answer.get("noul"))
                if answer.get("type") == "noul" and probability is not None:
                    answers.nouls[key] = NoulAnswer(probability=probability)
            elif answer.get("type") == "choice" and answer.get("choice"):
                names = names_by_key[key]
                answers.choices[key] = ChoiceAnswer(
                    choice=names.get(answer["choice"], answer["choice"]),
                    probabilities={
                        names.get(name, name): float(probability)
                        for name, probability in (
                            answer.get("probabilities") or {}
                        ).items()
                        if _probability(probability) is not None
                    },
                    confidence=_probability(answer.get("confidence")) or 0.0,
                )
        received = data.get("received")
        tokens = (
            received.get("input_tokens") if isinstance(received, dict) else None
        )
        cost = (
            self.calculate_cost(tokens)
            if isinstance(tokens, int) and not isinstance(tokens, bool)
            else None
        )
        return answers, cost


def _to_multipart(
    audio: bytes,
    questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
    context: Optional[str],
) -> Tuple[Multipart, Dict[str, Dict[str, str]]]:
    if not questions:
        raise DeepEvalError(
            "HumeDecisionModel.decide requires at least one question."
        )
    if len(questions) > HUME_MAX_QUESTIONS:
        raise DeepEvalError(
            f"Hume answers at most {HUME_MAX_QUESTIONS} questions "
            "per request."
        )
    wire_questions: Dict[str, Dict[str, Any]] = {}
    names_by_key: Dict[str, Dict[str, str]] = {}
    for key, question in questions.items():
        wire_questions[key], names_by_key[key] = _to_wire(question)
    request: Dict[str, Any] = {"questions": wire_questions}
    if context:
        request["context"] = _clipped(context, HUME_MAX_CONTEXT_CHARS)
    return (
        Multipart(
            file_field="file",
            filename="audio.wav",
            content=audio,
            content_type="audio/wav",
            fields={"request": request},
        ),
        names_by_key,
    )


def _to_wire(
    question: Union[NoulQuestion, ChoiceQuestion],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    instructions = _clipped(question.instructions, HUME_MAX_INSTRUCTIONS_CHARS)
    if isinstance(question, NoulQuestion):
        wire: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        criteria = {
            side: _clipped(value, HUME_MAX_CRITERION_CHARS)
            for side, value in (
                ("true", question.true),
                ("false", question.false),
            )
            if value is not None
        }
        if criteria:
            wire["criteria"] = criteria
        return wire, {}
    if isinstance(question, ChoiceQuestion):
        if not 2 <= len(question.options) <= HUME_MAX_CHOICES:
            raise DeepEvalError(
                f"Hume pick-one questions take 2 to {HUME_MAX_CHOICES} "
                f"options, not {len(question.options)}."
            )
        names: Dict[str, str] = {}
        options: List[Dict[str, str]] = []
        for option, description in question.options.items():
            name = _clipped(option, HUME_MAX_CHOICE_NAME_CHARS)
            suffix = 2
            while name in names:
                tail = f" ({suffix})"
                name = (
                    _clipped(option, HUME_MAX_CHOICE_NAME_CHARS - len(tail))
                    + tail
                )
                suffix += 1
            names[name] = option
            criterion = {"name": name}
            if description:
                criterion["description"] = _clipped(
                    description, HUME_MAX_CHOICE_DESCRIPTION_CHARS
                )
            options.append(criterion)
        return (
            {
                "type": "choice",
                "instructions": instructions,
                "criteria": options,
            },
            names,
        )
    raise DeepEvalError(
        f"Hume answers yes/no and pick-one questions, not {type(question).__name__}."
    )


def _checked(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise DeepEvalError("Hume returned no answers.")
    return data


def _probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1 else None
