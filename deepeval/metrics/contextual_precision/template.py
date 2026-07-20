from deepeval.metrics.prompt_template import BasePromptTemplate


class ContextualPrecisionTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`ContextualPrecisionMetric`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``ContextualPrecisionMetric(evaluation_template=YourTemplate)``. Any method
    you do not override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string):

    - ``generate_verdicts(input, expected_output, document_count_str,
      context_to_display, multimodal_note)``
    - ``generate_reason(input, verdicts, score)``

    The default template bodies live under
    ``deepeval/metrics/contextual_precision/templates/``.
    """
