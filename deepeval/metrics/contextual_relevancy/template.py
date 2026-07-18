from deepeval.metrics.prompt_template import BasePromptTemplate


class ContextualRelevancyTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`ContextualRelevancyMetric`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``ContextualRelevancyMetric(evaluation_template=YourTemplate)``. Any method
    you do not override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string).
    ``generate_verdicts`` also receives extra rendering keyword arguments, so
    declare ``**kwargs`` in your override to stay compatible:

    - ``generate_verdicts(input, context, **kwargs)``
    - ``generate_reason(input, irrelevant_statements, relevant_statements,
      score)``

    The default template bodies live under
    ``deepeval/metrics/contextual_relevancy/templates/``.
    """
