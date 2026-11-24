import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_qag_verdicts,
    a_generate_with_schema_and_extract,
    a_run_system_one_eval,
    construct_verbose_logs,
    generate_qag_verdicts,
    generate_with_schema_and_extract,
    initialize_model,
    initialize_system_one_model,
    parse_questions,
    prepare_measure,
    run_system_one_eval,
    score_qag_verdicts,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.metrics.transcription_accuracy.schema import (
    TranscriptionAccuracyScoreReason,
    TranscriptionAccuracyVerdict,
    Verdicts,
)
from deepeval.models import (
    DeepEvalBaseLLM,
    DeepEvalBaseSTT,
    DeepEvalBaseSystemOneModel,
    OpenAISTTModel,
)
from deepeval.models.speech_selection import initialize_stt_model
from deepeval.templates import make_template_class
from deepeval.test_case import (
    Audio,
    ConversationalTestCase,
    MultiTurnParams,
    Turn,
)
from deepeval.utils import get_or_create_event_loop, prettify_list


TranscriptionAccuracyTemplate = make_template_class(
    "TranscriptionAccuracyMetric"
)

# What the metric treats as the record of what was actually said: the
# caller's scripted `content` (voice simulations) or each turn's `audio`,
# re-transcribed (recorded calls).
TranscriptionComparison = Literal["content", "audio"]

# Without it, gpt-4o-transcribe cleans gibberish out of what it hears, so the
# judge never sees the noise it is meant to catch. whisper-1 is left out: on
# pure noise this prompt makes it invent speech.
VERBATIM_TRANSCRIPTION_PROMPT = (
    "Transcribe verbatim. Include every sound: fillers like um and uh, "
    "partial words, gibberish and non-words spelled phonetically, and mark "
    "noise like [beep] or [static]."
)


def _initialize_stt_model_for_audio(
    stt_model: Optional[Union[str, DeepEvalBaseSTT]],
) -> DeepEvalBaseSTT:
    if isinstance(stt_model, DeepEvalBaseSTT):
        return stt_model
    model = initialize_stt_model(stt_model)
    if isinstance(model, OpenAISTTModel) and model.name.startswith("gpt-4o"):
        model.transcription_kwargs = {
            "prompt": VERBATIM_TRANSCRIPTION_PROMPT,
            **model.transcription_kwargs,
        }
    return model


def get_transcribed_exchanges(turns: List[Turn]) -> List[Dict[str, str]]:
    exchanges: List[Dict[str, str]] = []
    spoken: List[str] = []
    for turn in turns:
        if turn.role == "user":
            if turn.content:
                spoken.append(turn.content)
            continue
        if turn.provider_transcript is not None and spoken:
            exchanges.append(
                {
                    "spoken": " ".join(spoken),
                    "transcribed": turn.provider_transcript,
                    "agent_reply": turn.content,
                }
            )
        spoken = []
    return exchanges


def _text_the_system_used(turns: List[Turn], index: int) -> str:
    turn = turns[index]
    if turn.role == "assistant":
        # Duplex simulations replace an agent turn's content with a
        # transcription of its clip and keep what it meant to say here.
        return (turn.metadata or {}).get("intended_content", turn.content)
    if turn.provider_transcript is not None:
        return turn.provider_transcript
    # The agent's reply carries its transcript of every caller turn since it
    # last spoke, so it stands in only when this turn is that whole stretch.
    before = turns[index - 1] if index > 0 else None
    after = turns[index + 1] if index + 1 < len(turns) else None
    if (
        (before is None or before.role == "assistant")
        and after is not None
        and after.role == "assistant"
        and after.provider_transcript is not None
    ):
        return after.provider_transcript
    return turn.content


def get_audio_exchanges(
    turns: List[Turn],
) -> List[Tuple[Audio, Dict[str, Any]]]:
    pairs: List[Tuple[Audio, Dict[str, Any]]] = []
    for index, turn in enumerate(turns):
        if turn.audio is None:
            continue
        pairs.append(
            (
                turn.audio,
                {
                    "turn": index,
                    "role": turn.role,
                    "spoken": None,
                    "transcribed": _text_the_system_used(turns, index),
                    "start_time": turn.audio.start_time,
                },
            )
        )
    return pairs


def _verdict_for_exchange(
    exchange: Dict[str, str], verdict: Verdict, probability: float
) -> TranscriptionAccuracyVerdict:
    if verdict == Verdict.NO:
        detail = (
            f"The caller said {exchange['spoken']!r} but the agent heard "
            f"{exchange['transcribed']!r}."
        )
    else:
        detail = f"The agent heard {exchange['spoken']!r} faithfully."
    return TranscriptionAccuracyVerdict(
        verdict=verdict, reason=f"{detail} (P(yes)={probability:.2f})"
    )


def _verdict_for_audio_exchange(
    exchange: Dict[str, Any], verdict: Verdict, probability: float
) -> TranscriptionAccuracyVerdict:
    who = "caller" if exchange["role"] == "user" else "agent"
    start_time = exchange.get("start_time")
    when = f" at {start_time:.1f}s" if start_time is not None else ""
    if verdict == Verdict.NO:
        detail = (
            f"The {who} was heard saying {exchange['spoken']!r}{when} but "
            f"the transcript has {exchange['transcribed']!r}."
        )
    else:
        detail = (
            f"The transcript accounts for what the {who} was heard "
            f"saying{when}."
        )
    return TranscriptionAccuracyVerdict(
        verdict=verdict, reason=f"{detail} (P(yes)={probability:.2f})"
    )


class TranscriptionAccuracyMetric(BaseConversationalMetric):
    _required_test_case_params = [
        MultiTurnParams.CONTENT,
        MultiTurnParams.ROLE,
        MultiTurnParams.PROVIDER_TRANSCRIPT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        compare_against: TranscriptionComparison = "content",
        stt_model: Optional[Union[str, DeepEvalBaseSTT]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            TranscriptionAccuracyTemplate
        ] = TranscriptionAccuracyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        if compare_against not in ("content", "audio"):
            raise ValueError(
                "compare_against must be 'content' or 'audio', got "
                f"{compare_against!r}."
            )
        if compare_against == "content" and stt_model is not None:
            raise ValueError(
                "`stt_model` is only used when compare_against='audio'."
            )
        self.compare_against = compare_against
        if compare_against == "audio":
            self._required_test_case_params = [
                MultiTurnParams.CONTENT,
                MultiTurnParams.ROLE,
                MultiTurnParams.AUDIO,
            ]
        self.stt_model = (
            _initialize_stt_model_for_audio(stt_model)
            if compare_against == "audio"
            else None
        )
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                if run_system_one_eval(self, test_case):
                    return self.score

                self.exchanges = self._get_exchanges(test_case)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=(
                        [f"STT model: {self.stt_model.get_model_name()}"]
                        if self.compare_against == "audio"
                        else []
                    )
                    + [
                        f"Exchanges:\n{prettify_list(self.exchanges)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            self.exchanges = await self._a_get_exchanges(test_case)
            self.verdicts = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=(
                    [f"STT model: {self.stt_model.get_model_name()}"]
                    if self.compare_against == "audio"
                    else []
                )
                + [
                    f"Exchanges:\n{prettify_list(self.exchanges)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _get_exchanges(
        self, test_case: ConversationalTestCase
    ) -> List[Dict[str, Any]]:
        if self.compare_against == "audio":
            return self._transcribe_exchanges(
                get_audio_exchanges(test_case.turns)
            )
        return get_transcribed_exchanges(test_case.turns)

    async def _a_get_exchanges(
        self, test_case: ConversationalTestCase
    ) -> List[Dict[str, Any]]:
        if self.compare_against == "audio":
            return await self._a_transcribe_exchanges(
                get_audio_exchanges(test_case.turns)
            )
        return get_transcribed_exchanges(test_case.turns)

    def _transcribe_exchanges(
        self, pairs: List[Tuple[Audio, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        exchanges = []
        for clip, exchange in pairs:
            spoken, cost = self.stt_model.transcribe(clip)
            if cost is not None:
                self._accrue_cost(cost)
            exchange["spoken"] = spoken
            exchanges.append(exchange)
        return exchanges

    async def _a_transcribe_exchanges(
        self, pairs: List[Tuple[Audio, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        results = await asyncio.gather(
            *(self.stt_model.a_transcribe(clip) for clip, _ in pairs)
        )
        exchanges = []
        for (_, exchange), (spoken, cost) in zip(pairs, results):
            if cost is not None:
                self._accrue_cost(cost)
            exchange["spoken"] = spoken
            exchanges.append(exchange)
        return exchanges

    def _verdicts_prompt(self) -> str:
        method = (
            "generate_audio_verdicts"
            if self.compare_against == "audio"
            else "generate_verdicts"
        )
        return self._get_prompt(method, exchanges=self.exchanges)

    async def _a_generate_verdicts(
        self,
    ) -> List[TranscriptionAccuracyVerdict]:
        if len(self.exchanges) == 0:
            return []

        return await a_generate_qag_verdicts(
            metric=self,
            prompt=self._verdicts_prompt(),
            verdict_cls=TranscriptionAccuracyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _generate_verdicts(self) -> List[TranscriptionAccuracyVerdict]:
        if len(self.exchanges) == 0:
            return []

        return generate_qag_verdicts(
            metric=self,
            prompt=self._verdicts_prompt(),
            verdict_cls=TranscriptionAccuracyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _experimental_system_one_spec(self) -> SystemOneVerdictSpec:
        if self.compare_against == "audio":
            return SystemOneVerdictSpec(
                instructions=self._get_prompt(
                    "_experimental_system_one_audio_verdict"
                ),
                items=self.exchanges,
                item_key="exchange",
                build_verdict=_verdict_for_audio_exchange,
            )
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=self.exchanges,
            item_key="exchange",
            build_verdict=_verdict_for_exchange,
        )

    def _system_one_eval_spec(
        self, test_case: ConversationalTestCase
    ) -> Optional[SystemOneEvalSpec]:
        method = (
            "_experimental_system_one_audio_questions"
            if self.compare_against == "audio"
            else "_experimental_system_one_questions"
        )
        return SystemOneEvalSpec(
            evaluation_params=self._required_test_case_params,
            questions=parse_questions(self._get_prompt(method)),
            extra_state={"exchanges": self._get_exchanges(test_case)},
        )

    def _mistranscriptions(self) -> List[str]:
        return [
            verdict.reason
            for verdict in self.verdicts
            if verdict is not None
            and verdict.verdict is not None
            and verdict.verdict == Verdict.NO
        ]

    def _reason_prompt(self) -> str:
        method = (
            "generate_audio_reason"
            if self.compare_against == "audio"
            else "generate_reason"
        )
        return self._get_prompt(
            method,
            score=self.score,
            mistranscriptions=self._mistranscriptions(),
        )

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(),
            schema_cls=TranscriptionAccuracyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        return generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(),
            schema_cls=TranscriptionAccuracyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(self) -> float:
        # None verdicts (failed generation / out-of-vocabulary replies) are
        # dropped by score_qag_verdicts.
        return score_qag_verdicts(self, self.verdicts, passing=(Verdict.YES,))

    @property
    def __name__(self):
        return "Transcription Accuracy"
