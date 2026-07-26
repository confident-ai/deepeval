from deepeval.metrics.prompt_template import BasePromptTemplate


class FaithfulnessTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`FaithfulnessMetric`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``FaithfulnessMetric(evaluation_template=YourTemplate)``. Any method you do
    not override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string):

    - ``generate_claims(actual_output, multimodal_instruction)``
    - ``generate_truths(retrieval_context, limit)``
    - ``generate_verdicts(claims, retrieval_context)``
    - ``generate_reason(contradictions, score)``

    The default template bodies live under
    ``deepeval/metrics/faithfulness/templates/``.
    """
