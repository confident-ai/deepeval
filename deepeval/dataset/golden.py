import json
import re
import warnings
from dataclasses import dataclass
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing import Any, Literal, Optional, Dict, List, Tuple
from typing import Union
from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.test_case import (
    ToolCall,
    Turn,
    MLLMImage,
    RetrievedContextData,
)
from deepeval.test_case.llm_test_case import _MLLM_IMAGE_REGISTRY
from .expectations import Expectations


Frequency = Literal["rare", "normal", "frequent"]
InterruptionLevel = Frequency
OverlapBehavior = Literal["yield", "adaptive", "insist"]
LineQuality = Literal["wideband", "phone", "mobile", "speakerphone"]
Pace = Literal["slow", "fast"]

_INTERRUPTION_LEVELS = ("rare", "normal", "frequent")
_OVERLAP_BEHAVIORS = ("yield", "adaptive", "insist")

_FREQUENCY_PROSE = {
    "rare": "You rarely cut in while the agent is still speaking.",
    "normal": "You sometimes cut in while the agent is still speaking.",
    "frequent": "You often cut in while the agent is still speaking.",
}
_OVERLAP_PROSE = {
    "yield": "If you both end up talking at once, you back off and let the agent finish.",
    "adaptive": "If you both end up talking at once, you hold on briefly and then back off.",
    "insist": "If you both end up talking at once, you keep talking until the agent stops.",
}
_PACE_PROSE = {
    "slow": "Speak very slowly, leaving a clear pause between phrases.",
    "fast": "Speak very fast, as quickly as you can while staying clear.",
}
# Closes every voice direction. Without it a TTS model drifts back to its own
# even, polished delivery within a sentence or two.
_HOLD_DELIVERY = (
    "Hold this delivery on every word, from the first to the last. Never "
    "fall back to a neutral, polished or customer-service voice. You are a "
    "real person on a phone call, not a narrator."
)


def _strength_band(strength: float) -> int:
    """Which third of the scale a strength falls in: 0, 1 or 2."""
    return min(2, int(strength * 3))


# A TTS model cannot follow a number, so each third of the scale gets its own
# wording, from barely there to as strong as it gets.
_ACCENT_PROSE = (
    "Speak with a light {accent} accent: clearly from there, but soft and "
    "easy on the ear.",
    "Speak with a strong {accent} accent on every word, the way someone "
    "who has lived there all their life speaks.",
    "Speak with an extremely thick, heavy {accent} accent on every word, "
    "so broad that an outsider has to listen closely to follow it.",
)
_EMOTION_PROSE = (
    "Sound a little {emotion}: it shows in your tone, but you are holding "
    "it in.",
    "Sound very {emotion}: let it colour your tone, pace and emphasis in "
    "every sentence.",
    "Sound overwhelmingly {emotion}, far past the point of hiding it: it "
    "takes over your voice, your breathing and your pace completely.",
)
_STRENGTH_WORD = ("light", "strong", "extreme")
_EMOTION_DEGREE = ("a little ", "", "very ")
_HOW_OFTEN = {
    "rare": "once in a while",
    "normal": "every few sentences",
    "frequent": "in almost every sentence, often more than once",
}
# The speech patterns that change what the caller says. The rest change how
# the audio is timed and mixed, which the words cannot carry.
_WORDING_PATTERNS = {
    "fillers": (
        "You use filler words such as um, uh, well or you know {often}."
    ),
    "trails_off": (
        "You leave a sentence unfinished {often}, ending it on a dash, as in "
        '"I was hoping to move it to -".'
    ),
    "asks_to_repeat": (
        "You ask the agent to repeat or rephrase what it just said {often}."
    ),
    "self_corrects": (
        "You say the wrong thing and correct yourself mid-sentence {often}, "
        'as in "Tuesday - no, sorry, Wednesday".'
    ),
}


@dataclass
class InterruptionBehavior:
    """How readily the simulated caller talks over the agent.

    Set on ``Persona(interruption_behavior=...)`` to enable duplex barge-in;
    leaving it ``None`` keeps the half-duplex turn-based path.

    ``frequency`` controls how readily the caller interrupts. ``overlap``
    controls whether they yield, adapt, or insist when both sides keep talking.
    Floor-control timing is selected internally as a coordinated preset.
    """

    frequency: InterruptionLevel = "normal"
    overlap: OverlapBehavior = "adaptive"

    def __post_init__(self) -> None:
        if self.frequency not in _INTERRUPTION_LEVELS:
            raise ValueError(
                f"Invalid interruption frequency {self.frequency!r}; expected "
                "'rare', 'normal', or 'frequent'."
            )
        if self.overlap not in _OVERLAP_BEHAVIORS:
            raise ValueError(
                f"Invalid overlap behavior {self.overlap!r}; expected "
                "'yield', 'adaptive', or 'insist'."
            )


class VoiceSettings(BaseModel):
    """How the simulated caller sounds."""

    # Voice ID passed to the TTS model.
    id: Optional[str] = Field(default=None)
    accent: Optional[str] = Field(default=None)
    # How thick the accent is, from a faint trace (0.0) to very heavy (1.0).
    accent_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    emotion: Optional[str] = Field(default=None)
    # How openly the emotion shows, from a hint (0.0) to overwhelming (1.0).
    emotion_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    pace: Optional[Pace] = Field(default=None)
    # How loud the caller is, where 1.0 is the voice as the TTS model made it.
    volume: float = Field(default=1.0, ge=0.0, le=1.0)
    # Anything else about delivery: energy, tone.
    instructions: Optional[str] = Field(default=None)


class LineSettings(BaseModel):
    """The connection the caller's speech travels over."""

    # "phone" band-limits and μ-law encodes the caller like a landline call,
    # "mobile" is a weak mobile connection, and "speakerphone" is a caller
    # talking from across a room.
    quality: LineQuality = Field(default="wideband")
    # Short gaps where the caller's audio cuts out.
    dropouts: Optional[Frequency] = Field(default=None)
    # Short spans where the caller's audio is distorted.
    garbled: Optional[Frequency] = Field(default=None)
    static: float = Field(default=0.0, ge=0.0, le=1.0)
    # Network delay added before each thing the caller says.
    delay_ms: int = Field(default=0, ge=0)


class BackgroundSpeaker(BaseModel):
    """Someone else in the room, heard faintly behind the caller."""

    # Who they are and what they talk about.
    characteristics: str
    voice: Optional[VoiceSettings] = Field(default=None)
    frequency: Frequency = Field(default="normal")
    volume: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("voice", mode="before")
    @classmethod
    def _voice_from_id(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value


class BackgroundNoiseSettings(BaseModel):
    """What can be heard around the simulated caller."""

    # Path or URL to a .wav or .mp3 file, looped under the caller's speech.
    audio: Optional[str] = Field(default=None)
    volume: float = Field(default=0.3, ge=0.0, le=1.0)
    other_speakers: List[BackgroundSpeaker] = Field(default_factory=list)


class SpeechPatternSettings(BaseModel):
    """Habits in how the simulated caller talks."""

    # --- change what the caller says; apply to text simulations too ---
    fillers: Optional[Frequency] = Field(default=None)
    trails_off: Optional[Frequency] = Field(default=None)
    asks_to_repeat: Optional[Frequency] = Field(default=None)
    self_corrects: Optional[Frequency] = Field(default=None)
    switches_language: Optional[Frequency] = Field(default=None)

    # --- voice only: change how the caller's audio is timed and mixed ---
    mid_sentence_pauses: Optional[Frequency] = Field(default=None)
    slow_to_answer: Optional[Frequency] = Field(default=None)
    # The caller turns away and says something to someone else in the room.
    side_speech: Optional[Frequency] = Field(default=None)

    @property
    def shapes_audio(self) -> bool:
        return any(
            (self.mid_sentence_pauses, self.slow_to_answer, self.side_speech)
        )


class Persona(BaseModel):
    """Who the simulated user is, how they behave, and how they sound.

    A persona pairs with a `ConversationalGolden`: the persona is *who* the
    user is, while the golden's `scenario` and `expected_outcome` are *what*
    they are trying to do. Keep behavioral traits here and task instructions
    on the golden.

    `name`, `characteristics`, `metadata`, `languages` and the wording half of
    `speech_patterns` apply to every simulation. Everything else is voice-only
    and is ignored by text simulations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = Field(default=None)
    # The free-form persona prompt: demographics, personality, emotional arc,
    # speaking style — who the user is, never what they want.
    # Successor to `ConversationalGolden.user_description`.
    characteristics: str
    metadata: Optional[Dict] = Field(default=None)
    languages: Optional[List[str]] = Field(default=None)
    speech_patterns: Optional[SpeechPatternSettings] = Field(default=None)

    # --- voice only: behavior ---
    interruption_behavior: Optional[InterruptionBehavior] = Field(default=None)
    speaks_first: bool = Field(default=True)
    muted: bool = Field(default=False)
    # Seconds of agent silence or hold music before the caller hangs up.
    hold_timeout: Optional[float] = Field(default=None, gt=0)

    # --- voice only: audio ---
    voice: Optional[VoiceSettings] = Field(default=None)
    line: Optional[LineSettings] = Field(default=None)
    background_noise: Optional[BackgroundNoiseSettings] = Field(default=None)
    # Speaks through this model instead of the `VoiceConfig` one.
    tts_model: Optional[DeepEvalBaseTTS] = Field(default=None, exclude=True)
    multilingual_stt: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def _nest_flat_voice_fields(cls, data: Any) -> Any:
        """Accept a bare voice ID and the flat names these settings once had."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        voice = data.get("voice")
        if isinstance(voice, str):
            voice = {"id": voice}
        if "voice_instructions" in data:
            instructions = data.pop("voice_instructions")
            if isinstance(voice, VoiceSettings):
                voice = voice.model_dump()
            voice = {**(voice or {}), "instructions": instructions}
        if voice is not None:
            data["voice"] = voice
        if "line_quality" in data:
            quality = data.pop("line_quality")
            line = data.get("line")
            if isinstance(line, LineSettings):
                line = line.model_dump()
            data["line"] = {**(line or {}), "quality": quality}
        return data

    @model_validator(mode="after")
    def _check_language_switching(self):
        patterns = self.speech_patterns
        if (
            patterns is not None
            and patterns.switches_language is not None
            and len(self.languages or []) < 2
        ):
            raise ValueError(
                "`switches_language` needs at least two `languages` on the "
                "persona to switch between."
            )
        return self

    def prompt_block(self) -> str:
        """Render the persona for a simulator prompt.

        Delimited rather than quoted because `characteristics` is free-form
        multi-line text that can itself contain quotes and tag blocks.
        """
        sections: List[str] = []
        if self.name:
            sections.append(f"Name: {self.name}")
        sections.append(self.characteristics.strip())
        if self.metadata:
            sections.append(
                "Details:\n" + json.dumps(self.metadata, indent=2, default=str)
            )

        behavior: List[str] = []
        if self.voice is not None and self.voice.emotion:
            degree = _EMOTION_DEGREE[
                _strength_band(self.voice.emotion_strength)
            ]
            behavior.append(f"Right now you feel {degree}{self.voice.emotion}.")
        if self.languages and len(self.languages) > 1:
            behavior.append("You speak " + ", ".join(self.languages) + ".")
        behavior.extend(self._speech_pattern_prose())
        if self.interruption_behavior is not None:
            behavior.append(
                _FREQUENCY_PROSE[self.interruption_behavior.frequency]
            )
            behavior.append(_OVERLAP_PROSE[self.interruption_behavior.overlap])
        if not self.speaks_first:
            behavior.append(
                "You wait for the agent to speak first and never open the "
                "conversation yourself."
            )
        if behavior:
            sections.append("\n".join(behavior))

        body = "\n\n".join(section for section in sections if section)
        return f"<persona>\n{body}\n</persona>"

    def voice_brief(self) -> str:
        """The voice settings in words, for whoever writes the direction."""
        voice = self.voice
        if voice is None:
            return "None"
        stated = [
            (
                f"Accent: {voice.accent}, "
                f"{_STRENGTH_WORD[_strength_band(voice.accent_strength)]}"
                if voice.accent
                else ""
            ),
            (
                f"Emotion: {voice.emotion}, "
                f"{_STRENGTH_WORD[_strength_band(voice.emotion_strength)]}"
                if voice.emotion
                else ""
            ),
            f"Pace: {voice.pace}" if voice.pace else "",
            f"Also: {voice.instructions}" if voice.instructions else "",
        ]
        return "\n".join(line for line in stated if line) or "None"

    def _speech_pattern_prose(self) -> List[str]:
        patterns = self.speech_patterns
        if patterns is None:
            return []
        prose = [
            sentence.format(often=_HOW_OFTEN[getattr(patterns, name)])
            for name, sentence in _WORDING_PATTERNS.items()
            if getattr(patterns, name) is not None
        ]
        if patterns.switches_language is not None:
            prose.append(
                "You switch between your languages mid-sentence "
                f"{_HOW_OFTEN[patterns.switches_language]}."
            )
        return prose

    def delivery_instructions(
        self, derived: Optional[str] = None
    ) -> Optional[str]:
        """The caller's voice direction: one instruction, the same on every line.

        `derived` is the direction the simulator model wrote from the persona
        before the call. The explicit voice settings follow it so they have
        the last word, and the whole is closed by the reminder to hold it.
        """
        voice = self.voice
        parts = [derived.strip()] if derived and derived.strip() else []
        if voice is not None:
            if voice.accent:
                parts.append(
                    _ACCENT_PROSE[_strength_band(voice.accent_strength)].format(
                        accent=voice.accent
                    )
                )
            if voice.emotion:
                parts.append(
                    _EMOTION_PROSE[
                        _strength_band(voice.emotion_strength)
                    ].format(emotion=voice.emotion)
                )
            if voice.pace:
                parts.append(_PACE_PROSE[voice.pace])
            if voice.instructions:
                parts.append(voice.instructions.strip())
        if not parts:
            return None
        return " ".join(parts + [_HOLD_DELIVERY])

    def tts_kwargs(
        self,
        tts_model: Optional[DeepEvalBaseTTS] = None,
        delivery: Optional[str] = None,
    ) -> Dict:
        """Synthesis kwargs for this persona's voice on `tts_model`.

        The voice ID is the only vocal knob declared on
        `DeepEvalBaseTTS.a_synthesize`, so it is the only one portable across
        TTS implementations. Delivery is passed as `instructions` only to
        models that take it, and is ignored by the rest.
        """
        voice = self.voice
        kwargs: Dict = {}
        if voice is not None and voice.id is not None:
            kwargs["voice"] = voice.id
        instructions = self.delivery_instructions(delivery)
        if not instructions:
            return kwargs
        if tts_model is not None and tts_model.supports_voice_instructions():
            kwargs["instructions"] = instructions
        elif self.delivery_instructions() is not None:
            warnings.warn(
                "Persona `voice` accent, emotion and instructions are "
                "ignored: the TTS model does not take instructions."
            )
        return kwargs

    def caller_tts_model(self, tts_model: DeepEvalBaseTTS) -> DeepEvalBaseTTS:
        """The TTS model this persona speaks through on one call."""
        from deepeval.voice.caller_speech import CallerSpeechTTS
        from deepeval.voice.phone_line import PhoneLineTTS

        tts_model = self.tts_model or tts_model
        if self.line is not None and self.line.quality != "wideband":
            tts_model = PhoneLineTTS(tts_model, self.line.quality)
        if CallerSpeechTTS.applies_to(self):
            tts_model = CallerSpeechTTS(tts_model, self)
        return tts_model


def resolve_persona(
    persona: Optional[Persona], user_description: Optional[str]
) -> Tuple[Optional[Persona], Optional[str]]:
    """Reconcile the `persona` field with the deprecated `user_description`.

    Both are kept in sync so `userDescription` keeps flowing to Confident AI
    and through CSV/JSON datasets while `user_description` is being retired.
    """
    if persona is None:
        if user_description is None:
            return None, None
        warnings.warn(
            "'user_description' is deprecated and will be removed in a future "
            "release. Use 'persona=Persona(characteristics=...)' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return Persona(characteristics=user_description), user_description
    if (
        user_description is not None
        and user_description != persona.characteristics
    ):
        raise ValueError(
            "Pass either 'persona' or the deprecated 'user_description', not "
            "both with conflicting text."
        )
    return persona, persona.characteristics


class Golden(BaseModel):
    id: Optional[str] = Field(default=None, exclude=True)
    input: str
    actual_output: Optional[str] = Field(
        default=None, serialization_alias="actualOutput"
    )
    expected_output: Optional[str] = Field(
        default=None, serialization_alias="expectedOutput"
    )
    context: Optional[List[str]] = Field(default=None)
    retrieval_context: Optional[List[Union[str, RetrievedContextData]]] = Field(
        default=None, serialization_alias="retrievalContext"
    )
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    tools_called: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="expectedTools"
    )
    token_cost: Optional[float] = Field(
        default=None,
        serialization_alias="tokenCost",
        validation_alias=AliasChoices("tokenCost", "token_cost"),
    )
    input_token_count: Optional[int] = Field(
        default=None,
        serialization_alias="inputTokenCount",
        validation_alias=AliasChoices("inputTokenCount", "input_token_count"),
    )
    output_token_count: Optional[int] = Field(
        default=None,
        serialization_alias="outputTokenCount",
        validation_alias=AliasChoices("outputTokenCount", "output_token_count"),
    )
    source_file: Optional[str] = Field(
        default=None, serialization_alias="sourceFile"
    )
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    expectations: Optional[Expectations] = Field(default=None)
    expected_labels: Optional[Dict[str, str]] = Field(
        default=None,
        serialization_alias="expectedLabels",
        validation_alias=AliasChoices("expectedLabels", "expected_labels"),
    )
    multimodal: bool = Field(False, exclude=True)
    images_mapping: Dict[str, MLLMImage] = Field(
        default=None, alias="imagesMapping"
    )
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)
    _dataset_version: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]"
        auto_detect = (
            any(
                [
                    re.search(pattern, self.input or "") is not None,
                    re.search(pattern, self.actual_output or "") is not None,
                ]
            )
            if isinstance(self.input, str)
            else self.multimodal
        )
        if self.retrieval_context is not None:
            auto_detect = auto_detect or any(
                re.search(
                    pattern,
                    c.context if isinstance(c, RetrievedContextData) else c,
                )
                is not None
                for c in self.retrieval_context
                if isinstance(c, (RetrievedContextData, str))
            )
        if self.context is not None:
            auto_detect = auto_detect or any(
                re.search(pattern, context) is not None
                for context in self.context
            )

        self.multimodal = auto_detect

        return self

    def _get_images_mapping(self) -> Dict[str, MLLMImage]:
        pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]"
        image_ids = set()

        def extract_ids_from_string(s: Optional[str]) -> None:
            """Helper to extract image IDs from a string."""
            if s is not None and isinstance(s, str):
                matches = re.findall(pattern, s)
                image_ids.update(matches)

        def extract_ids_from_list(lst: Optional[List[str]]) -> None:
            """Helper to extract image IDs from a list of strings."""
            if lst is not None:
                for item in lst:
                    extract_ids_from_string(item)

        extract_ids_from_string(self.input)
        extract_ids_from_string(self.actual_output)
        extract_ids_from_string(self.expected_output)
        extract_ids_from_list(self.context)
        if self.retrieval_context:
            extract_ids_from_list(
                [
                    c.context if isinstance(c, RetrievedContextData) else c
                    for c in self.retrieval_context
                ]
            )

        images_mapping = {}
        for img_id in image_ids:
            if img_id in _MLLM_IMAGE_REGISTRY:
                images_mapping[img_id] = _MLLM_IMAGE_REGISTRY[img_id]

        return images_mapping if len(images_mapping) > 0 else None

    def _prepare_for_api(self) -> None:
        self.name = None
        self.images_mapping = self._get_images_mapping()
        if self.retrieval_context:
            self.retrieval_context = [
                rc.context if hasattr(rc, "context") else rc
                for rc in self.retrieval_context
            ]


class ConversationalGolden(BaseModel):
    id: Optional[str] = Field(default=None, exclude=True)
    scenario: str
    expected_outcome: Optional[str] = Field(
        None, serialization_alias="expectedOutcome"
    )
    persona: Optional[Persona] = Field(default=None, exclude=True)
    # Deprecated: use `persona`. Kept in sync with `persona.characteristics`.
    user_description: Optional[str] = Field(
        None, serialization_alias="userDescription"
    )
    context: Optional[List[str]] = Field(default=None)
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    expectations: Optional[Expectations] = Field(default=None)
    expected_labels: Optional[Dict[str, str]] = Field(
        default=None,
        serialization_alias="expectedLabels",
        validation_alias=AliasChoices("expectedLabels", "expected_labels"),
    )
    turns: Optional[List[Turn]] = Field(default=None)
    multimodal: bool = Field(False, exclude=True)
    images_mapping: Dict[str, MLLMImage] = Field(
        default=None, alias="imagesMapping"
    )
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)
    _dataset_version: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def sync_persona(self):
        self.persona, self.user_description = resolve_persona(
            self.persona, self.user_description
        )
        return self

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]"
        if self.scenario:
            if re.search(pattern, self.scenario) is not None:
                self.multimodal = True
                return self
        if self.expected_outcome:
            if re.search(pattern, self.expected_outcome) is not None:
                self.multimodal = True
                return self
        if self.user_description:
            if re.search(pattern, self.user_description) is not None:
                self.multimodal = True
                return self
        if self.context and any(re.search(pattern, c) for c in self.context):
            self.multimodal = True
            return self
        if self.turns:
            for turn in self.turns:
                if re.search(pattern, turn.content) is not None:
                    self.multimodal = True
                    return self
                if turn.retrieval_context is not None:
                    self.multimodal = self.multimodal or any(
                        re.search(
                            pattern,
                            (
                                c.context
                                if isinstance(c, RetrievedContextData)
                                else c
                            ),
                        )
                        is not None
                        for c in turn.retrieval_context
                        if isinstance(c, (RetrievedContextData, str))
                    )

        return self

    def _get_images_mapping(self) -> Dict[str, MLLMImage]:
        pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]"
        image_ids = set()

        def extract_ids_from_string(s: Optional[str]) -> None:
            """Helper to extract image IDs from a string."""
            if s is not None and isinstance(s, str):
                matches = re.findall(pattern, s)
                image_ids.update(matches)

        def extract_ids_from_list(lst: Optional[List[str]]) -> None:
            """Helper to extract image IDs from a list of strings."""
            if lst is not None:
                for item in lst:
                    extract_ids_from_string(item)

        extract_ids_from_string(self.scenario)
        extract_ids_from_string(self.expected_outcome)
        extract_ids_from_list(self.context)
        extract_ids_from_string(self.user_description)
        if self.turns:
            for turn in self.turns:
                extract_ids_from_string(turn.content)
                if turn.retrieval_context:
                    extract_ids_from_list(
                        [
                            (
                                c.context
                                if isinstance(c, RetrievedContextData)
                                else c
                            )
                            for c in turn.retrieval_context
                        ]
                    )

        images_mapping = {}
        for img_id in image_ids:
            if img_id in _MLLM_IMAGE_REGISTRY:
                images_mapping[img_id] = _MLLM_IMAGE_REGISTRY[img_id]

        return images_mapping if len(images_mapping) > 0 else None

    def _prepare_for_api(self) -> None:
        self.name = None
        self.images_mapping = self._get_images_mapping()
        if self.turns:
            for turn in self.turns:
                if turn.retrieval_context:
                    turn.retrieval_context = [
                        rc.context if hasattr(rc, "context") else rc
                        for rc in turn.retrieval_context
                    ]
