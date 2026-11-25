import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.models.llms.multimodal import to_openai_content
from deepeval.models.llms.openai_model import OpenAIModel, retry_openai
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.system_one.constants import (
    DEFAULT_OPENAI_AUDIO_DECISION_MODEL,
)
from deepeval.models.system_one.schema import (
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    SystemOneAnswers,
)
from deepeval.models.utils import prompt_audio_tokens
from deepeval.test_case import Audio

# Without it, OpenAI's audio models mostly reply that they can't listen to audio.
AUDIO_DECISION_SYSTEM_PROMPT = (
    "You analyze audio recordings of phone calls for a quality-assurance "
    "tool. You can hear the attached audio clip. Answer only with JSON."
)


class AudioVerdict(BaseModel):
    answer: Literal["yes", "no"]
    confidence: float = Field(ge=0, le=1)


class OpenAIAudioDecisionModel(DeepEvalBaseAudioDecisionModel):
    """Answers yes/no questions about a WAV recording with an OpenAI audio
    model, which says yes or no and how sure it is of that answer."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(model or DEFAULT_OPENAI_AUDIO_DECISION_MODEL)

    def load_model(self) -> OpenAIModel:
        return OpenAIModel(
            model=self.name, api_key=self.api_key, base_url=self.base_url
        )

    def get_model_name(self, *args, **kwargs) -> str:
        return f"OpenAI {self.name}"

    def decide(
        self,
        audio: bytes,
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        context: Optional[str] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        clip = _clip(audio)
        return _answers(
            questions,
            [
                self._verdict(_messages(question, context, clip))
                for question in _noul_questions(questions)
            ],
        )

    async def a_decide(
        self,
        audio: bytes,
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
        context: Optional[str] = None,
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        clip = _clip(audio)
        return _answers(
            questions,
            await asyncio.gather(
                *(
                    self._a_verdict(_messages(question, context, clip))
                    for question in _noul_questions(questions)
                )
            ),
        )

    @retry_openai
    def _verdict(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[AudioVerdict, Optional[float]]:
        client = self.model.load_model(async_mode=False)
        return self._parsed(
            client.chat.completions.create(**self._request(messages))
        )

    @retry_openai
    async def _a_verdict(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[AudioVerdict, Optional[float]]:
        client = self.model.load_model(async_mode=True)
        return self._parsed(
            await client.chat.completions.create(**self._request(messages))
        )

    def _request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "model": self.name,
            "messages": messages,
            "modalities": ["text"],
            **self.model._temperature_kwargs(),
            **self.model.generation_kwargs,
        }

    def _parsed(
        self, completion: ChatCompletion
    ) -> Tuple[AudioVerdict, Optional[float]]:
        usage = completion.usage
        cost = self.model.calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            prompt_audio_tokens(usage),
        )
        verdict = AudioVerdict.model_validate(
            trim_and_load_json(completion.choices[0].message.content)
        )
        return verdict, cost


def _clip(audio: bytes) -> List[Audio]:
    return [Audio.from_bytes(audio, "audio/wav")]


def _noul_questions(
    questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
) -> List[NoulQuestion]:
    if not questions:
        raise DeepEvalError(
            "OpenAIAudioDecisionModel.decide requires at least one question."
        )
    for question in questions.values():
        if not isinstance(question, NoulQuestion):
            raise DeepEvalError(
                "OpenAIAudioDecisionModel answers yes/no questions only, not "
                f"{type(question).__name__}."
            )
    return list(questions.values())


def _messages(
    question: NoulQuestion, context: Optional[str], clip: List[Audio]
) -> List[Dict[str, Any]]:
    content = to_openai_content(_prompt(question, context), clip)
    return [
        {"role": "system", "content": AUDIO_DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": sorted(
                content, key=lambda part: part["type"] != "input_audio"
            ),
        },
    ]


def _prompt(question: NoulQuestion, context: Optional[str]) -> str:
    lines = [context] if context else []
    lines += [
        "Listen to the audio and answer this yes/no question about it.",
        f"Question: {question.instructions}",
    ]
    if question.true is not None:
        lines.append(f'Answer "yes" when: {question.true}')
    if question.false is not None:
        lines.append(f'Answer "no" when: {question.false}')
    lines.append(
        'Reply with JSON only: {"answer": "yes" or "no", "confidence": a '
        "number from 0 to 1 for how sure you are of that answer}."
    )
    return "\n".join(lines)


def _answers(
    questions: Dict[str, Union[NoulQuestion, ChoiceQuestion]],
    replies: List[Tuple[AudioVerdict, Optional[float]]],
) -> Tuple[SystemOneAnswers, Optional[float]]:
    answers = SystemOneAnswers()
    for key, (verdict, _) in zip(questions, replies):
        answers.nouls[key] = NoulAnswer(
            probability=(
                verdict.confidence
                if verdict.answer == "yes"
                else 1 - verdict.confidence
            )
        )
    costs = [cost for _, cost in replies if cost is not None]
    return answers, sum(costs) if costs else None
