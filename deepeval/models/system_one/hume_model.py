import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    SystemOneAnswers,
)
from deepeval.models.utils import require_secret_api_key

HUME_API_URL = "https://api.cloud.hume.ai/v0"
HUME_DECISION_MODEL = "audio-decisions"
HUME_COST_PER_INPUT_TOKEN = 0.47 / 1_000_000
HUME_MAX_QUESTIONS = 20
HUME_MAX_INSTRUCTIONS_CHARS = 500
HUME_MAX_CRITERION_CHARS = 300
HUME_MAX_CHOICE_NAME_CHARS = 50
HUME_MAX_CHOICE_DESCRIPTION_CHARS = 100
HUME_MAX_CONTEXT_CHARS = 1_000
HUME_RETRY_STATUSES = {429, 500, 503}
HUME_MAX_RETRY_WAIT_SECONDS = 5.0
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
        **kwargs,
    ):
        self.api_key: Optional[SecretStr] = (
            SecretStr(api_key)
            if api_key is not None
            else get_settings().HUME_API_KEY
        )
        self.kwargs = kwargs
        super().__init__(model or HUME_DECISION_MODEL)

    def load_model(self, *args, **kwargs) -> Any:
        return None

    def get_model_name(self, *args, **kwargs) -> str:
        return f"Hume {self.name}"

    def decide(
        self,
        audio: bytes,
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        context: Optional[str] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        if not questions:
            raise DeepEvalError(
                "HumeDecisionModel.decide requires at least one question."
            )
        if len(questions) > HUME_MAX_QUESTIONS:
            raise DeepEvalError(
                f"Hume answers at most {HUME_MAX_QUESTIONS} questions per "
                "request."
            )
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Hume",
            env_var_name="HUME_API_KEY",
            param_hint="`api_key` to HumeDecisionModel(...)",
        )
        wire_questions: Dict[str, Dict[str, Any]] = {}
        names_by_key: Dict[str, Dict[str, str]] = {}
        for key, question in questions.items():
            wire_questions[key], names_by_key[key] = _to_wire(question)
        request: Dict[str, Any] = {"questions": wire_questions}
        if context:
            request["context"] = _clipped(context, HUME_MAX_CONTEXT_CHARS)
        data = _post(api_key, audio, json.dumps(request))
        return _from_wire(data, questions, names_by_key)


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


def _post(api_key: str, audio: bytes, request: str) -> Dict[str, Any]:
    def send() -> requests.Response:
        return requests.post(
            f"{HUME_API_URL}/decisions/file",
            headers={"X-Hume-Api-Key": api_key},
            files={
                "request": (None, request, "application/json"),
                "file": ("audio.wav", audio, "audio/wav"),
            },
            timeout=HUME_TIMEOUT_SECONDS,
        )

    response = send()
    if response.status_code in HUME_RETRY_STATUSES:
        time.sleep(_retry_wait(response))
        response = send()
    if response.status_code != 200:
        raise DeepEvalError(
            f"Hume audio decisions failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    try:
        data = response.json()
    except ValueError as error:
        raise DeepEvalError(
            "Hume returned a response that is not JSON."
        ) from error
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise DeepEvalError("Hume returned no answers.")
    return data


def _retry_wait(response: requests.Response) -> float:
    try:
        wait = float(response.headers.get("Retry-After", "1"))
    except ValueError:
        wait = 1.0
    return min(max(wait, 0.0), HUME_MAX_RETRY_WAIT_SECONDS)


def _from_wire(
    data: Dict[str, Any],
    questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
    names_by_key: Dict[str, Dict[str, str]],
) -> Tuple[SystemOneAnswers, Optional[float]]:
    answers = SystemOneAnswers()
    for key, question in questions.items():
        answer = data["answers"].get(key)
        if not isinstance(answer, dict):
            continue
        if isinstance(question, NoulQuestion) and answer.get("type") == "noul":
            answers.nouls[key] = NoulAnswer(probability=float(answer["noul"]))
        elif answer.get("type") == "choice":
            names = names_by_key[key]
            answers.choices[key] = ChoiceAnswer(
                choice=names.get(answer["choice"], answer["choice"]),
                probabilities={
                    names.get(name, name): float(probability)
                    for name, probability in (
                        answer.get("probabilities") or {}
                    ).items()
                },
                confidence=float(answer.get("confidence") or 0.0),
            )
    received = data.get("received")
    tokens = (
        received.get("input_tokens") if isinstance(received, dict) else None
    )
    cost = (
        tokens * HUME_COST_PER_INPUT_TOKEN if isinstance(tokens, int) else None
    )
    return answers, cost
