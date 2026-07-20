from deepeval.metrics.prompt_template import BasePromptTemplate


class ContextualRecallTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`ContextualRecallMetric`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``ContextualRecallMetric(evaluation_template=YourTemplate)``. Any method you
    do not override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string). These also
    receive extra rendering keyword arguments (e.g. ``content_type``,
    ``context_to_display``), so declare ``**kwargs`` in your override to stay
    compatible:

    - ``generate_verdicts(expected_output, **kwargs)``
    - ``generate_reason(expected_output, supportive_reasons,
      unsupportive_reasons, score, **kwargs)``

    The default template bodies live under
    ``deepeval/metrics/contextual_recall/templates/``.
    """
