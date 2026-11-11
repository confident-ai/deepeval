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
from deepeval.constants import SUPPORTED_PROVIDER_SLUGS, slugify
from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.test_case import (
    ToolCall,
    Turn,
    MLLMImage,
    RetrievedContextData,
)
from deepeval.test_case.llm_test_case import _MLLM_IMAGE_REGISTRY
from .expectations import Expectations


InterruptionLevel = Literal["rare", "normal", "frequent"]
OverlapBehavior = Literal["yield", "adaptive", "insist"]
LineQuality = Literal["wideband", "phone"]

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


class BackgroundNoiseSettings(BaseModel):
    """Ambient audio looped underneath the simulated caller's speech."""

    # Path to a .wav or .mp3 file.
    audio: str
    volume: float = Field(default=0.3, ge=0.0, le=1.0)


class Persona(BaseModel):
    """Who the simulated user is, how they behave, and how they sound.

    A persona pairs with a `ConversationalGolden`: the persona is *who* the
    user is, while the golden's `scenario` and `expected_outcome` are *what*
    they are trying to do. Keep behavioral traits here and task instructions
    on the golden.

    `name`, `characteristics` and `metadata` apply to every simulation. Every
    field below `metadata` is voice-only and is ignored by text simulations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = Field(default=None)
    # The free-form persona prompt: demographics, personality, emotional arc,
    # speaking style, filler words — who the user is, never what they want.
    # Successor to `ConversationalGolden.user_description`.
    characteristics: str
    metadata: Optional[Dict] = Field(default=None)

    # --- voice only: behavior ---
    interruption_behavior: Optional[InterruptionBehavior] = Field(default=None)
    speaks_first: bool = Field(default=True)
    muted: bool = Field(default=False)

    # --- voice only: audio ---
    tts_model: Optional[Union[str, DeepEvalBaseTTS]] = Field(default=None)
    voice: Optional[Union[str, Dict[str, str]]] = Field(default=None)
    voice_instructions: Optional[str] = Field(default=None)
    background_noise: Optional[BackgroundNoiseSettings] = Field(default=None)
    line_quality: LineQuality = Field(default="wideband")
    multilingual_stt: bool = Field(default=False)
    # Seconds of agent silence or hold music before the caller hangs up.
    hold_timeout: Optional[float] = Field(default=None, gt=0)

    @field_validator("tts_model", mode="plain")
    @classmethod
    def _accept_any_tts_model(cls, model: Any) -> Any:
        if (
            model is None
            or isinstance(model, str)
            or hasattr(model, "a_synthesize")
        ):
            return model
        raise ValueError(
            "tts_model must be a TTS model name or a DeepEvalBaseTTS, not "
            f"{type(model).__name__}."
        )

    @field_validator("voice")
    @classmethod
    def _key_voices_by_provider(
        cls, voice: Optional[Union[str, Dict[str, str]]]
    ) -> Optional[Union[str, Dict[str, str]]]:
        if not isinstance(voice, dict):
            return voice
        keyed = {
            slugify(provider): voice_id for provider, voice_id in voice.items()
        }
        unknown = sorted(set(keyed) - SUPPORTED_PROVIDER_SLUGS)
        if unknown:
            raise ValueError(
                f"Unknown voice provider {unknown[0]!r}; key voices by "
                "provider, e.g. 'openai', 'elevenlabs', 'cartesia' or "
                "'deepgram'."
            )
        return keyed

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

    def tts_kwargs(self, tts_model: Optional[DeepEvalBaseTTS] = None) -> Dict:
        """Synthesis kwargs for this persona's voice on `tts_model`.

        `voice` is the only vocal knob declared on
        `DeepEvalBaseTTS.a_synthesize`, so it goes to every TTS model; when it
        is keyed by provider, the model gets its own provider's entry, or its
        default voice if there is none. `voice_instructions` go out as
        `instructions` only to a model whose `supports_voice_instructions()`
        says it takes them: other providers pass unknown kwargs straight to
        their APIs, which reject them.
        """
        voice = self._voice_on(tts_model)
        kwargs: Dict = {} if voice is None else {"voice": voice}
        # Duck-typed TTS models need not define the capability check.
        supports_instructions = getattr(
            tts_model, "supports_voice_instructions", None
        )
        if (
            self.voice_instructions
            and supports_instructions is not None
            and supports_instructions()
        ):
            kwargs["instructions"] = self.voice_instructions
        return kwargs

    def _voice_on(self, tts_model: Optional[DeepEvalBaseTTS]) -> Optional[str]:
        if not isinstance(self.voice, dict):
            return self.voice
        provider = getattr(tts_model, "provider", None)
        return None if provider is None else self.voice.get(slugify(provider))


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
