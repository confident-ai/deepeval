"""HTTP adapter for endpoints implementing the typed System One contract."""

import asyncio
import hashlib
import json
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import aiohttp
import requests
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.limits import check_context_budget
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
from deepeval.models.utils import EvaluationCost


def _number(value: Any, label: str, maximum: Optional[float] = None) -> float:
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not finite
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise DeepEvalError(f"Invalid System One {label}.")
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeepEvalError(
                "System One response contains duplicate JSON keys."
            )
        result[key] = value
    return result


class SystemOneAPIModel(DeepEvalBaseSystemOneModel):
    """A compatible decision endpoint, not a generic chat-completions model.

    Prices are dollars per token. Unknown prices or usage produce unknown
    cost, never an assumed Jev price. Token budgets are optional estimates;
    the server must enforce its actual tokenizer limits.
    """

    def __init__(
        self,
        model: Optional[str],
        base_url: Optional[str],
        *,
        endpoint: str = "/v1/systemone",
        api_key: Optional[Union[str, SecretStr]] = None,
        auth: Optional[str] = None,
        timeout: float = 30.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        question_types: Sequence[str] = ("noul", "choice", "score"),
        max_state_tokens: Optional[int] = None,
        max_request_tokens: Optional[int] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise DeepEvalError("Set DEEPEVAL_SYSTEM_ONE_MODEL or pass model.")
        parsed = urlsplit(base_url or "")
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise DeepEvalError(
                "System One base_url must be an HTTP(S) URL without credentials, query, or fragment."
            )
        path = urlsplit(endpoint)
        if (
            not endpoint.startswith("/")
            or endpoint.startswith("//")
            or path.scheme
            or path.netloc
            or path.query
            or path.fragment
        ):
            raise DeepEvalError(
                "System One endpoint must be an absolute URL path."
            )
        self.url = base_url.rstrip("/") + endpoint
        self.api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(api_key or "")
        )
        if any(
            ord(char) < 32 or ord(char) > 126
            for char in self.api_key.get_secret_value()
        ):
            raise DeepEvalError("System One API key must contain printable ASCII.")
        self.auth = (
            auth
            if auth is not None
            else ("bearer" if self.api_key.get_secret_value() else "none")
        )
        if self.auth not in ("none", "bearer", "api_key"):
            raise DeepEvalError(
                "System One auth must be none, bearer, or api_key."
            )
        if self.auth != "none" and not self.api_key.get_secret_value():
            raise DeepEvalError(
                "Set DEEPEVAL_SYSTEM_ONE_API_KEY for authenticated System One requests."
            )
        if self.auth == "none" and self.api_key.get_secret_value():
            raise DeepEvalError(
                "System One auth=none cannot be combined with an API key."
            )
        self.timeout = _number(timeout, "timeout")
        if self.timeout == 0:
            raise DeepEvalError("System One timeout must be positive.")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_state_tokens", max_state_tokens),
            ("max_request_tokens", max_request_tokens),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise DeepEvalError(
                    f"System One {label} must be a positive integer."
                )
        if max_response_bytes is None:
            raise DeepEvalError("System One max_response_bytes is required.")
        self.max_response_bytes = max_response_bytes
        self.max_state_tokens = max_state_tokens
        self.max_request_tokens = max_request_tokens
        self.question_types = frozenset(question_types)
        if not self.question_types or not self.question_types <= {
            "noul",
            "choice",
            "score",
        }:
            raise DeepEvalError(
                "System One question_types must contain noul, choice, or score."
            )
        self.input_price = (
            None
            if cost_per_input_token is None
            else _number(cost_per_input_token, "input price")
        )
        self.output_price = (
            None
            if cost_per_output_token is None
            else _number(cost_per_output_token, "output price")
        )
        identity = json.dumps(
            {
                "url": self.url,
                "auth": self.auth,
                "key": self.api_key.get_secret_value(),
                "timeout": self.timeout,
                "question_types": sorted(self.question_types),
                "state_limit": max_state_tokens,
                "request_limit": max_request_tokens,
                "response_limit": max_response_bytes,
                "input_price": self.input_price,
                "output_price": self.output_price,
            },
            sort_keys=True,
        )
        self._identity = hashlib.sha256(identity.encode()).hexdigest()
        super().__init__(model)
        # Endpoint model IDs may contain namespaces (for example org/model).
        self.name = model

    def load_model(self):
        # Clients are request-scoped so no session is shared across event loops.
        return self.url

    def get_model_name(self):
        return f"{self.name} (System One API: {self._identity})"

    def _questions(
        self, questions: Dict[str, SystemOneQuestion]
    ) -> Dict[str, Any]:
        if not questions:
            raise DeepEvalError("System One requires at least one question.")
        result = {}
        for key, question in questions.items():
            if not isinstance(key, str) or not key:
                raise DeepEvalError(
                    "System One question IDs must be nonempty strings."
                )
            if isinstance(question, NoulQuestion):
                kind = "noul"
                criteria = (
                    {"true": question.true, "false": question.false}
                    if question.true is not None or question.false is not None
                    else None
                )
            elif isinstance(question, ChoiceQuestion):
                kind, criteria = "choice", question.options
                if len(criteria) < 2:
                    raise DeepEvalError(
                        "System One choice requires at least two options."
                    )
            elif isinstance(question, ScoreQuestion):
                kind, criteria = "score", question.levels
                if len(criteria) < 2:
                    raise DeepEvalError(
                        "System One score requires at least two levels."
                    )
            else:
                raise DeepEvalError("Unsupported System One question type.")
            if kind not in self.question_types:
                raise DeepEvalError(
                    f"System One provider does not support {kind} questions."
                )
            result[key] = {"type": kind, "instructions": question.instructions}
            if criteria is not None:
                result[key]["criteria"] = criteria
        return result

    def check_request(self, state, questions):
        wire = self._questions(questions)
        check_context_budget(
            state,
            wire,
            max_state_tokens=self.max_state_tokens,
            max_request_tokens=self.max_request_tokens,
        )

    def _payload(self, state, questions):
        self.check_request(state, questions)
        try:
            return json.dumps(
                {
                    "model": self.name,
                    "state": state,
                    "questions": self._questions(questions),
                },
                allow_nan=False,
            ).encode()
        except (ValueError, TypeError):
            raise DeepEvalError(
                "System One state and questions must be JSON serializable with finite numbers."
            ) from None

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth == "bearer":
            headers["Authorization"] = (
                "Bearer " + self.api_key.get_secret_value()
            )
        elif self.auth == "api_key":
            headers["X-API-Key"] = self.api_key.get_secret_value()
        return headers

    def _check_status(self, status):
        if not 200 <= status < 300:
            # Never include response bodies, URLs, credentials, or submitted data.
            raise DeepEvalError(
                f"System One endpoint returned HTTP {status}; no fallback was attempted."
            )

    def _append(self, buffer, chunk):
        if len(buffer) + len(chunk) > self.max_response_bytes:
            raise DeepEvalError(
                "System One response exceeded max_response_bytes."
            )
        buffer.extend(chunk)

    def decide(
        self, state, questions
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        payload = self._payload(state, questions)
        try:
            with requests.Session() as client:
                client.trust_env = False
                with client.post(
                    self.url,
                    data=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                ) as response:
                    self._check_status(response.status_code)
                    body = bytearray()
                    for chunk in response.iter_content(65536):
                        self._append(body, chunk)
        except requests.RequestException:
            raise DeepEvalError(
                "System One HTTP request failed; check endpoint connectivity and timeout."
            ) from None
        return self._parse(body, questions)

    async def a_decide(
        self, state, questions
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        payload = self._payload(state, questions)
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,
            ) as client:
                async with client.post(
                    self.url,
                    data=payload,
                    headers=self._headers(),
                    allow_redirects=False,
                ) as response:
                    self._check_status(response.status)
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        self._append(body, chunk)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise DeepEvalError(
                "System One HTTP request failed; check endpoint connectivity and timeout."
            ) from None
        return self._parse(body, questions)

    def _parse(self, body, questions):
        try:
            data = json.loads(body, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError):
            raise DeepEvalError(
                "System One endpoint returned invalid JSON."
            ) from None
        if (
            not isinstance(data, dict)
            or data.get("truncated", False) is not False
        ):
            raise DeepEvalError(
                "System One response must be an object without truncation."
            )
        state_usage = data.get("usage")
        if isinstance(state_usage, dict):
            total = state_usage.get("state_tokens")
            used = state_usage.get("state_tokens_used")
            for value in (total, used):
                if value is not None and (type(value) is not int or value < 0):
                    raise DeepEvalError(
                        "System One state token counts must be nonnegative integers."
                    )
            if total is not None and used is not None and total != used:
                raise DeepEvalError(
                    "System One response indicates truncated state."
                )
        raw = data.get("answers")
        if not isinstance(raw, dict) or set(raw) != set(questions):
            raise DeepEvalError(
                "System One answer IDs must exactly match the question IDs."
            )
        result = SystemOneAnswers()
        for key, question in questions.items():
            answer = raw[key]
            kind = self._questions({key: question})[key]["type"]
            if not isinstance(answer, dict) or answer.get("type") != kind:
                raise DeepEvalError(
                    "System One answer type does not match its question."
                )
            if kind == "noul":
                result.nouls[key] = NoulAnswer(
                    probability=_number(
                        answer.get("noul"), "noul probability", 1
                    )
                )
                continue
            expected = (
                set(question.options)
                if kind == "choice"
                else {str(i) for i in range(len(question.levels))}
            )
            probabilities = answer.get("probabilities")
            if (
                not isinstance(probabilities, dict)
                or set(probabilities) != expected
            ):
                raise DeepEvalError(
                    "System One probabilities must cover exactly the question options or level indices."
                )
            probabilities = {
                k: _number(p, "probability", 1)
                for k, p in probabilities.items()
            }
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
                raise DeepEvalError("System One probabilities must sum to one.")
            confidence = _number(answer.get("confidence"), "confidence", 1)
            if kind == "choice":
                choice = answer.get("choice")
                if (
                    not isinstance(choice, str)
                    or choice not in expected
                    or probabilities[choice] < max(probabilities.values())
                ):
                    raise DeepEvalError(
                        "System One choice must be a highest-probability option."
                    )
                result.choices[key] = ChoiceAnswer(
                    choice=choice,
                    probabilities=probabilities,
                    confidence=confidence,
                )
            else:
                score = _number(
                    answer.get("score"), "score", len(question.levels) - 1
                )
                distribution = {int(k): p for k, p in probabilities.items()}
                mean = sum(k * p for k, p in distribution.items())
                if not math.isclose(score, mean, abs_tol=1e-5):
                    raise DeepEvalError(
                        "System One score must match its probability-weighted level index."
                    )
                result.scores[key] = ScoreAnswer(
                    score=score,
                    probabilities=distribution,
                    confidence=confidence,
                )
        return result, self._cost(data.get("usage"))

    def _cost(self, usage):
        if usage is None:
            return None
        if not isinstance(usage, dict):
            raise DeepEvalError("Invalid System One usage.")
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise DeepEvalError(
                    "System One usage counts must be nonnegative integers."
                )
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        if (
            self.input_price is None
            or self.output_price is None
            or inputs is None
            or outputs is None
        ):
            return None
        cost = inputs * self.input_price + outputs * self.output_price
        if not math.isfinite(cost):
            raise DeepEvalError("System One calculated cost is not finite.")
        return EvaluationCost(cost, inputs, outputs)
