import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from deepeval.errors import MissingTestCaseParamsError
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
from deepeval.metrics.voice.speech_quality.transcription_accuracy.utils import (
    word_errors,
)
from deepeval.metrics.voice.speech_quality.transcription_accuracy.schema import (
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

# Microsoft's speech-to-text guidance rates a 20% word error rate acceptable,
# 5-10% good and 30% or more poor.
DEFAULT_MAX_WORD_ERROR_PERCENT = 20

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


def _exchange_detail(exchange: Dict[str, str], verdict: Verdict) -> str:
    if verdict == Verdict.NO:
        return (
            f"The caller said {exchange['spoken']!r} but the agent heard "
            f"{exchange['transcribed']!r}."
        )
    return f"The agent heard {exchange['spoken']!r} faithfully."


def _audio_exchange_detail(exchange: Dict[str, Any], verdict: Verdict) -> str:
    who = "caller" if exchange["role"] == "user" else "agent"
    start_time = exchange.get("start_time")
    when = f" at {start_time:.1f}s" if start_time is not None else ""
    if verdict == Verdict.NO:
        return (
            f"The {who} was heard saying {exchange['spoken']!r}{when} but "
            f"the transcript has {exchange['transcribed']!r}."
        )
    return (
        f"The transcript accounts for what the {who} was heard "
        f"saying{when}."
    )


def _verdict_for_exchange(
    exchange: Dict[str, str], verdict: Verdict, probability: float
) -> TranscriptionAccuracyVerdict:
    return TranscriptionAccuracyVerdict(
        verdict=verdict,
        reason=f"{_exchange_detail(exchange, verdict)} "
        f"(P(yes)={probability:.2f})",
    )


def _verdict_for_audio_exchange(
    exchange: Dict[str, Any], verdict: Verdict, probability: float
) -> TranscriptionAccuracyVerdict:
    return TranscriptionAccuracyVerdict(
        verdict=verdict,
        reason=f"{_audio_exchange_detail(exchange, verdict)} "
        f"(P(yes)={probability:.2f})",
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
        use_llm: bool = True,
        max_word_error_percent: Optional[int] = None,
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
        if not use_llm and (
            model is not None
            or system_one_model is not None
            or eval_mode is not None
        ):
            raise ValueError(
                "`model`, `system_one_model` and `eval_mode` pick the judge, "
                "so they are only used when use_llm=True."
            )
        if use_llm and max_word_error_percent is not None:
            raise ValueError(
                "`max_word_error_percent` is only used when use_llm=False."
            )
        if max_word_error_percent is not None and max_word_error_percent < 0:
            raise ValueError("max_word_error_percent must be 0 or more.")
        self.use_llm = use_llm
        self.max_word_error_percent = (
            DEFAULT_MAX_WORD_ERROR_PERCENT
            if max_word_error_percent is None
            else max_word_error_percent
        )
        if use_llm:
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
        else:
            # Nothing is judged by a model, so cost starts at zero and only
            # speech-to-text, under compare_against="audio", adds to it.
            self.eval_mode = None
            self.model = None
            self.using_native_model = True
            self.system_one_model = None
            self.evaluation_model = None
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
                if self.use_llm and run_system_one_eval(self, test_case):
                    return self.score

                self.exchanges = self._require_exchanges(
                    self._get_exchanges(test_case)
                )
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
            if self.use_llm and await a_run_system_one_eval(self, test_case):
                return self.score

            self.exchanges = self._require_exchanges(
                await self._a_get_exchanges(test_case)
            )
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

    def _require_exchanges(
        self, exchanges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if exchanges:
            return exchanges
        if self.compare_against == "audio":
            error = (
                "No turn with audio could be compared with the call's "
                "transcript, so there is nothing for the "
                f"'{self.__name__}' metric to judge."
            )
        else:
            error = (
                "No assistant turn has a 'provider_transcript' that follows "
                "something the caller said, so there is nothing for the "
                f"'{self.__name__}' metric to compare."
            )
        self.error = error
        raise MissingTestCaseParamsError(error)

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
        if not self.use_llm:
            return self._word_error_verdicts()
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
        if not self.use_llm:
            return self._word_error_verdicts()
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
        if not self.use_llm:
            return self._word_error_reason()

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
        if not self.use_llm:
            return self._word_error_reason()

        return generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(),
            schema_cls=TranscriptionAccuracyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _word_error_verdicts(
        self,
    ) -> List[Optional[TranscriptionAccuracyVerdict]]:
        detail = (
            _audio_exchange_detail
            if self.compare_against == "audio"
            else _exchange_detail
        )
        verdicts: List[Optional[TranscriptionAccuracyVerdict]] = []
        total_errors = total_words = lines_over_limit = 0
        for exchange in self.exchanges:
            errors, words = word_errors(
                exchange["spoken"] or "", exchange["transcribed"] or ""
            )
            if words == 0:
                # Nothing was said, so there is no word to mishear.
                verdicts.append(None)
                continue
            percent = errors / words * 100
            verdict = (
                Verdict.YES
                if percent <= self.max_word_error_percent
                else Verdict.NO
            )
            lines_over_limit += verdict == Verdict.NO
            total_errors += errors
            total_words += words
            verdicts.append(
                TranscriptionAccuracyVerdict(
                    verdict=verdict,
                    reason=f"{detail(exchange, verdict)} ({errors} of "
                    f"{words} words wrong, {percent:.0f}%)",
                )
            )
        self.score_breakdown = {
            "word_error_percent": (
                round(total_errors / total_words * 100, 1)
                if total_words
                else None
            ),
            "lines": sum(verdict is not None for verdict in verdicts),
            "lines_over_limit": lines_over_limit,
        }
        return verdicts

    def _word_error_reason(self) -> str:
        breakdown = self.score_breakdown
        if not breakdown["lines"]:
            return "No line had spoken words to compare with its transcript."
        over = (
            f"{breakdown['lines_over_limit']} of {breakdown['lines']} lines"
            if breakdown["lines_over_limit"]
            else "No line"
        )
        return (
            f"{over} had over {self.max_word_error_percent}% of words "
            f"misheard; {breakdown['word_error_percent']:g}% of words were "
            "misheard overall."
        )

    def _calculate_score(self) -> float:
        # None verdicts (failed generation / out-of-vocabulary replies) are
        # dropped by score_qag_verdicts.
        return score_qag_verdicts(self, self.verdicts, passing=(Verdict.YES,))

    @property
    def __name__(self):
        return "Transcription Accuracy"
