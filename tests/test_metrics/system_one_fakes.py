"""Test doubles for System One (Jev, Drex) and the evaluation LLM.

Shared by the JevEval / DrexEval tests and the eval-mode tests so no test
needs a network or an API key. ``FakeDrex`` stands in for the Drex API at the
HTTP layer, under the real ``nace-sdk`` client."""

import base64
import json
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import httpx

from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneAnswers,
)
from deepeval.test_case import MLLMImage


class FakeSystemOneModel(DeepEvalBaseSystemOneModel):
    """Returns canned answers and records what it was asked.

    ``answers`` may be one ``SystemOneAnswers`` (returned on every call) or a
    list, consumed one per call. When ``answer_fn`` is given it is called
    with the questions and returns the answers, which lets a test answer
    however many Noul questions a metric happens to ask."""

    def __init__(
        self,
        answers: Any = None,
        cost: Optional[float] = 0.0,
        answer_fn=None,
        name: str = "fake-jev",
    ):
        self._name = name
        super().__init__(name)
        self.answers = answers
        self.cost = cost
        self.answer_fn = answer_fn
        self.calls: List[Tuple[Any, Dict[str, Any]]] = []

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return self._name

    def _next(self, questions: Dict[str, Any]) -> SystemOneAnswers:
        if self.answer_fn is not None:
            return self.answer_fn(questions)
        if isinstance(self.answers, list):
            return self.answers.pop(0)
        return self.answers

    def decide(
        self, state: Any, questions: Dict[str, Any]
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        self.calls.append((state, questions))
        return self._next(questions), self.cost

    async def a_decide(
        self, state: Any, questions: Dict[str, Any]
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        return self.decide(state, questions)


class ExplodingSystemOneModel(FakeSystemOneModel):
    """Raises ``exc`` on every call: simulates a Jev outage, an auth failure,
    a context overflow (pass a ``SystemOneContextLimitError``)..."""

    def __init__(self, exc: BaseException, name: str = "exploding-jev"):
        super().__init__(SystemOneAnswers(), name=name)
        self.exc = exc

    def decide(self, state, questions):
        self.calls.append((state, questions))
        raise self.exc

    async def a_decide(self, state, questions):
        return self.decide(state, questions)


def noul_answers(probability: float):
    """``answer_fn`` answering every Noul question with the same P(yes)."""

    def answer(questions: Dict[str, Any]) -> SystemOneAnswers:
        return SystemOneAnswers(
            nouls={
                key: NoulAnswer(probability=probability) for key in questions
            }
        )

    return answer


def answer_everything(probability: float = 0.9, confidence: float = 0.8):
    """``answer_fn`` answering any mix of questions: every Noul with
    ``probability``, every Score at its top level, every Choice with its
    first option. For tests that don't care how many questions a metric
    asks or of which type."""
    from deepeval.models.system_one.schema import (
        ChoiceQuestion,
        NoulQuestion,
        ScoreQuestion,
    )

    def answer(questions: Dict[str, Any]) -> SystemOneAnswers:
        out = SystemOneAnswers()
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                out.nouls[key] = NoulAnswer(probability=probability)
            elif isinstance(question, ScoreQuestion):
                top = len(question.levels) - 1
                out.scores[key] = ScoreAnswer(
                    score=top,
                    probabilities={i: float(i == top) for i in range(top + 1)},
                    confidence=confidence,
                )
            elif isinstance(question, ChoiceQuestion):
                first = next(iter(question.options))
                out.choices[key] = ChoiceAnswer(
                    choice=first,
                    probabilities={first: 1.0},
                    confidence=confidence,
                )
        return out

    return answer


def choice_answer(
    key: str, choice: str, probabilities: Dict[str, float], confidence: float
) -> SystemOneAnswers:
    return SystemOneAnswers(
        choices={
            key: ChoiceAnswer(
                choice=choice,
                probabilities=probabilities,
                confidence=confidence,
            )
        }
    )


def score_answer(
    key: str, score: float, probabilities: Dict[int, float], confidence: float
) -> SystemOneAnswers:
    return SystemOneAnswers(
        scores={
            key: ScoreAnswer(
                score=score, probabilities=probabilities, confidence=confidence
            )
        }
    )


class CannedLLM(DeepEvalBaseLLM):
    """Returns ``reply`` to every prompt and records the prompts. Used as the
    evaluation LLM for the legacy chain (extraction, verdicts, reasons)."""

    def __init__(self, reply: str = "{}", name: str = "canned-llm"):
        self._name = name
        super().__init__(name)
        self.reply = reply
        self.prompts: List[Any] = []

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return self._name

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.reply

    async def a_generate(self, prompt, *args, **kwargs):
        return self.generate(prompt)


class ScriptedLLM(CannedLLM):
    """Answers each prompt from ``replies`` in order (last one repeats)."""

    def __init__(self, replies: List[str], name: str = "scripted-llm"):
        super().__init__(replies[-1], name=name)
        self.replies = list(replies)

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class ExplodingLLM(DeepEvalBaseLLM):
    """Fails if any LLM call is made."""

    def __init__(self):
        super().__init__("exploding-llm")

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "exploding-llm"

    def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called")


###############################################
# Drex at the HTTP layer
###############################################


def solid_png(rgb, size=8) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


RED, BLUE = solid_png((255, 0, 0)), solid_png((0, 0, 255))


def image(data: bytes) -> MLLMImage:
    return MLLMImage(
        dataBase64=base64.b64encode(data).decode(), mimeType="image/png"
    )


def _answer(question: Dict[str, Any]) -> Dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.9}
    if question["type"] == "choice":
        labels = list(question["criteria"])
        rest = 0.2 / max(len(labels) - 1, 1)
        return {
            "type": "choice",
            "choice": labels[0],
            "confidence": 0.75,
            "probabilities": {
                label: 0.8 if i == 0 else rest for i, label in enumerate(labels)
            },
        }
    levels = question["criteria"]
    top = len(levels) - 1
    return {
        "type": "score",
        "score": float(top),
        "confidence": 0.7,
        "legend": levels,
        "probabilities": {str(i): float(i == top) for i in range(top + 1)},
    }


class FakeDrex:
    """Answers each question by type, like Drex; ``replies`` are returned
    first, one per request (``None`` means answer normally)."""

    def __init__(self, replies: Optional[List[Any]] = None):
        self.replies = list(replies or [])
        self.requests: List[httpx.Request] = []

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.replies:
            reply = self.replies.pop(0)
            if reply is not None:
                return reply
        body = json.loads(request.content)
        images = len(body.get("images") or [])
        return httpx.Response(
            200,
            headers={"x-request-id": "req_test"},
            json={
                "model": body["model"],
                "answers": {
                    key: _answer(q) for key, q in body["questions"].items()
                },
                "usage": {
                    "input_tokens": 1000,
                    "image_tokens": 64 * images or None,
                    "output_tokens": 10,
                },
                "evaluation_time_ms": 12.0,
                "request_id": "req_test",
            },
        )


def drex_error(
    status: int, message: str, headers=None, **fields
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers or {},
        json={
            "error": {"type": "error", "message": message, **fields},
            "request_id": "req_err",
        },
    )
