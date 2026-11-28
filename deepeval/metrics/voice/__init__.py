import importlib
import sys

from deepeval.metrics.voice.audio_defects import (
    AbruptCutoffMetric,
    AudioClippingMetric,
    AudioDropoutMetric,
    AudioGlitchMetric,
    AudioGlitchTemplate,
    AudioIntegrityMetric,
    AudioLoopMetric,
    DeadStreamMetric,
    EchoMetric,
    HeldToneMetric,
)
from deepeval.metrics.voice.audio_level import (
    BackgroundNoiseMetric,
    LoudnessMetric,
    LoudnessConsistencyMetric,
    VolumeVarianceMetric,
)
from deepeval.metrics.voice.prosody import (
    AbruptPitchJumpMetric,
    PausesMetric,
    PitchConsistencyMetric,
    PitchVariationMetric,
    SpeakingRateMetric,
)
from deepeval.metrics.voice.timing import (
    AgentInterruptionMetric,
    ResponseDelayConsistencyMetric,
    ResponseLatencyMetric,
    TimeToFirstResponseMetric,
    TurnTakingNaturalnessMetric,
    UserInterruptionMetric,
)
from deepeval.metrics.voice.responsiveness import (
    AgentResponsivenessMetric,
    FailedResponseMetric,
    RepromptMetric,
    SilentReplyMetric,
    UnexpectedCallEndMetric,
    VoiceReliabilityMetric,
)
from deepeval.metrics.voice.speech_quality import (
    SpeechIntelligibilityMetric,
    TranscriptionAccuracyMetric,
    TranscriptionAccuracyTemplate,
    VoiceConsistencyMetric,
    VoiceNaturalnessMetric,
)

# Paths these modules were released under before the voice metrics were
# grouped into folders here; imports from them keep working.
_RELEASED_PATHS = {
    "voice.consistency": "speech_quality.voice_consistency.voice_consistency",
    "voice.integrity": "audio_defects.audio_integrity.audio_integrity",
    "voice.intelligibility": "speech_quality.speech_intelligibility.speech_intelligibility",
    "voice.naturalness": "speech_quality.voice_naturalness.voice_naturalness",
    "voice.reliability": "responsiveness.voice_reliability.voice_reliability",
    "voice.responsiveness": "responsiveness.agent_responsiveness.agent_responsiveness",
    "voice.turn_taking": "timing.turn_taking_naturalness.turn_taking_naturalness",
    "audio_glitch": "audio_defects.audio_glitch",
    "audio_glitch.audio_glitch": "audio_defects.audio_glitch.audio_glitch",
    "audio_glitch.schema": "audio_defects.audio_glitch.schema",
    "transcription_accuracy": "speech_quality.transcription_accuracy",
    "transcription_accuracy.transcription_accuracy": "speech_quality.transcription_accuracy.transcription_accuracy",
    "transcription_accuracy.schema": "speech_quality.transcription_accuracy.schema",
}
for _old, _new in _RELEASED_PATHS.items():
    sys.modules.setdefault(
        f"deepeval.metrics.{_old}",
        importlib.import_module(f"deepeval.metrics.voice.{_new}"),
    )

__all__ = [
    "AbruptCutoffMetric",
    "AudioClippingMetric",
    "AudioDropoutMetric",
    "AudioGlitchMetric",
    "AudioGlitchTemplate",
    "AudioIntegrityMetric",
    "AudioLoopMetric",
    "DeadStreamMetric",
    "EchoMetric",
    "HeldToneMetric",
    "BackgroundNoiseMetric",
    "LoudnessMetric",
    "LoudnessConsistencyMetric",
    "VolumeVarianceMetric",
    "AbruptPitchJumpMetric",
    "PausesMetric",
    "PitchConsistencyMetric",
    "PitchVariationMetric",
    "SpeakingRateMetric",
    "AgentInterruptionMetric",
    "ResponseDelayConsistencyMetric",
    "ResponseLatencyMetric",
    "TimeToFirstResponseMetric",
    "TurnTakingNaturalnessMetric",
    "UserInterruptionMetric",
    "AgentResponsivenessMetric",
    "FailedResponseMetric",
    "RepromptMetric",
    "SilentReplyMetric",
    "UnexpectedCallEndMetric",
    "VoiceReliabilityMetric",
    "SpeechIntelligibilityMetric",
    "TranscriptionAccuracyMetric",
    "TranscriptionAccuracyTemplate",
    "VoiceConsistencyMetric",
    "VoiceNaturalnessMetric",
]
