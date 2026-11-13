import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.system_one.constants import (
    DEFAULT_OPENAI_DECISION_MODEL,
    OPENAI_DECISION_MODELS_DATA,
    OPENAI_DECISIONS_MAX_QUESTIONS,
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

retry_openai = create_retry_decorator(PS.OPENAI)

DECISIONS_PATH = "/decisions"

# OpenAI caps an input_json part at 256 KiB of compact UTF-8, 10,000 values
# and 32 levels of nesting; larger state goes as text instead.
_MAX_INPUT_JSON_BYTES = 256 * 1024
_MAX_INPUT_JSON_VALUES = 10_000
_MAX_INPUT_JSON_DEPTH = 32


def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _fits_input_json(state: Any) -> bool:
    values = 0
    stack = [(state, 1)]
    while stack:
        value, depth = stack.pop()
        values += 1
        if values > _MAX_INPUT_JSON_VALUES:
            return False
        if isinstance(value, (dict, list)):
            if depth > _MAX_INPUT_JSON_DEPTH:
                return False
            children = value.values() if isinstance(value, dict) else value
            stack.extend((child, depth + 1) for child in children)
    return True


def _to_input(state: Any) -> Any:
    if isinstance(state, (dict, list)):
        compact = json.dumps(
            state, ensure_ascii=False, separators=(",", ":"), default=str
        )
        if len(
            compact.encode("utf-8")
        ) <= _MAX_INPUT_JSON_BYTES and _fits_input_json(state):
            return [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_json", "data": state}],
                }
            ]
        return compact
    return _as_text(state)


class OpenAIDecisionModel(DeepEvalBaseSystemOneModel):
    """OpenAI's Decisions API (`POST /v1/decisions`) as a System One model."""

    # OpenAI publishes no context limit for its decision models; the API
    # rejects oversized requests itself.
    max_state_tokens: Optional[int] = None
    max_request_tokens: Optional[int] = None

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        **kwargs,
    ):
        settings = get_settings()

        if api_key is not None:
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.OPENAI_API_KEY

        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        model = model or DEFAULT_OPENAI_DECISION_MODEL

        if cost_per_input_token is not None and cost_per_input_token < 0:
            raise DeepEvalError("cost_per_input_token must be >= 0.")
        self.model_data = OPENAI_DECISION_MODELS_DATA.get(model)
        if cost_per_input_token is not None:
            self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = 0.0

        self.kwargs = kwargs
        self._async_client = None
        super().__init__(model)

    ###############################################
    # Decide
    ###############################################

    def decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        request_input = _to_input(state)
        client = self.load_model()
        responses = [
            self._post(client, request_input, chunk)
            for chunk in self._to_api_question_chunks(questions)
        ]
        return self._from_api_responses(responses, questions)

    async def a_decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        request_input = _to_input(state)
        client = self.load_model(async_mode=True)
        responses = await asyncio.gather(
            *(
                self._a_post(client, request_input, chunk)
                for chunk in self._to_api_question_chunks(questions)
            )
        )
        return self._from_api_responses(list(responses), questions)

    @retry_openai
    def _post(
        self, client: OpenAI, request_input: Any, questions: List[Dict]
    ) -> Dict:
        return client.post(
            DECISIONS_PATH,
            body=self._body(request_input, questions),
            cast_to=object,
        )

    @retry_openai
    async def _a_post(
        self, client: AsyncOpenAI, request_input: Any, questions: List[Dict]
    ) -> Dict:
        return await client.post(
            DECISIONS_PATH,
            body=self._body(request_input, questions),
            cast_to=object,
        )

    def _body(self, request_input: Any, questions: List[Dict]) -> Dict:
        return {"model": self.name, "input": request_input, "questions": questions}

    ###############################################
    # Translation
    ###############################################

    def _to_api_question_chunks(
        self, questions: Dict[str, SystemOneQuestion]
    ) -> List[List[Dict]]:
        if not questions:
            raise DeepEvalError(
                "OpenAIDecisionModel.decide requires at least one question."
            )
        api_questions = [
            self._to_api_question(key, question)
            for key, question in questions.items()
        ]
        return [
            api_questions[i : i + OPENAI_DECISIONS_MAX_QUESTIONS]
            for i in range(0, len(api_questions), OPENAI_DECISIONS_MAX_QUESTIONS)
        ]

    def _to_api_question(self, key: str, question: SystemOneQuestion) -> Dict:
        if isinstance(question, NoulQuestion):
            instructions = _as_text(question.instructions)
            if question.true is not None:
                instructions += f"\n\nTrue when: {_as_text(question.true)}"
            if question.false is not None:
                instructions += f"\nFalse when: {_as_text(question.false)}"
            return {
                "type": "predicate",
                "name": key,
                "instructions": instructions,
            }
        if isinstance(question, ChoiceQuestion):
            return {
                "type": "choice",
                "name": key,
                "instructions": _as_text(question.instructions),
                "choices": [
                    {"value": value, "description": _as_text(description)}
                    if description is not None
                    else {"value": value}
                    for value, description in question.options.items()
                ],
            }
        if isinstance(question, ScoreQuestion):
            return {
                "type": "score",
                "name": key,
                "instructions": _as_text(question.instructions),
                "levels": [
                    {"label": f"level_{index}", "description": _as_text(level)}
                    for index, level in enumerate(question.levels)
                ],
            }
        raise DeepEvalError(
            f"Unsupported System One question type: {type(question)}"
        )

    def _from_api_responses(
        self,
        responses: List[Dict],
        questions: Dict[str, SystemOneQuestion],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        answers = SystemOneAnswers()
        input_tokens = 0
        output_tokens = 0
        for response in responses:
            for answer in response.get("answers") or []:
                self._add_answer(answers, answer)
            usage = response.get("usage") or {}
            input_tokens += usage.get("input_tokens") or 0
            output_tokens += usage.get("output_tokens") or 0
        answered = (
            set(answers.nouls) | set(answers.choices) | set(answers.scores)
        )
        missing = [key for key in questions if key not in answered]
        if missing:
            raise DeepEvalError(
                f"OpenAI returned no answer for question(s): {', '.join(missing)}"
            )
        return answers, self.calculate_cost(input_tokens, output_tokens)

    def _add_answer(self, answers: SystemOneAnswers, answer: Dict) -> None:
        name = answer.get("name")
        kind = answer.get("type")
        if kind == "refusal":
            raise DeepEvalError(f"OpenAI refused to answer question '{name}'.")
        if kind == "predicate":
            answers.nouls[name] = NoulAnswer(probability=answer["probability"])
        elif kind == "choice":
            answers.choices[name] = ChoiceAnswer(
                choice=str(answer["choice"]),
                probabilities={
                    str(p["value"]): p["probability"]
                    for p in answer.get("probabilities") or []
                },
                confidence=answer["confidence"],
            )
        elif kind == "score":
            answers.scores[name] = ScoreAnswer(
                score=answer["score"],
                probabilities={
                    int(p["value"]): p["probability"]
                    for p in answer.get("probabilities") or []
                },
                confidence=answer["confidence"],
            )
        else:
            raise DeepEvalError(f"Unexpected OpenAI decision answer: {answer!r}")

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        if self.model_data.input_price is None:
            return None
        return EvaluationCost(
            input_tokens * self.model_data.input_price,
            input_tokens,
            output_tokens,
        )

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        if self._async_client is None:
            self._async_client = self._build_client(AsyncOpenAI)
        return self._async_client

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to OpenAIDecisionModel(...)",
        )
        kw = dict(api_key=api_key, base_url=self.base_url, **self.kwargs)
        if not sdk_retries_for(PS.OPENAI):
            kw["max_retries"] = 0
        if not kw.get("timeout"):
            kw["timeout"] = _request_timeout_seconds()
        return cls(**kw)

    def get_model_name(self):
        return f"{self.name} (OpenAI)"
