from typing import List, Optional, Sequence, Type, Union

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
        name: str = "tone_adherence",
        labels: Optional[Sequence[Union[str, Label]]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        self.tone = tone
        super().__init__(
            name=name,
            labels=labels if labels is not None else _default_labels(tone=tone),
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
            evaluation_template=evaluation_template,
        )
