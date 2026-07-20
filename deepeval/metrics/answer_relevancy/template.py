from deepeval.metrics.prompt_template import BasePromptTemplate


class AnswerRelevancyTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`AnswerRelevancyMetric`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``AnswerRelevancyMetric(evaluation_template=YourTemplate)``. Any method you do
    not override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string):

    - ``generate_statements(actual_output)``
    - ``generate_verdicts(input, statements)``
    - ``generate_reason(irrelevant_statements, input, score)``

    The default template bodies live under
    ``deepeval/metrics/answer_relevancy/templates/``.
    """
