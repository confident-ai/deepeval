from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM


def _default_labels(tone: Optional[str]) -> List[Label]:
    voice = (
        f"the configured tone ({tone})"
        if tone
        else "the configured tone of the assistant"
    )
    return [
        Label(
            "on_tone",
            f"The response matches {voice} in wording, register, and length.",
        ),
        Label(
            "off_tone",
            f"The response departs from {voice} in wording, register, or length.",
        ),
    ]


class ToneAdherenceClassifier(Classifier):
    """Labels whether the assistant matched the configured tone or voice."""

    def __init__(
        self,
        tone: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        self.tone = tone
        super().__init__(
            name="tone_adherence",
            labels=_default_labels(tone=tone),
            model=model,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
