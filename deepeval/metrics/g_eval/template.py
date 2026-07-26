from deepeval.metrics.prompt_template import BasePromptTemplate


class GEvalTemplate(BasePromptTemplate):
    """Customizable prompt template for :class:`GEval`.

    Subclass this and override any of the static methods below to customize the
    metric's prompts, then pass the subclass via
    ``GEval(evaluation_template=YourTemplate, ...)``. Any method you do not
    override falls back to deepeval's default template.

    Overridable methods (each must return the final prompt string). Because these
    receive several keyword arguments, declaring ``**kwargs`` in your override
    keeps it forward-compatible:

    - ``generate_evaluation_steps(criteria, parameters)``
    - ``generate_evaluation_results(evaluation_steps, test_case_content,
      parameters, rubric, score_range, _additional_context)``
    - ``generate_strict_evaluation_results(evaluation_steps, test_case_content,
      parameters, _additional_context)``

    The default template bodies live under
    ``deepeval/metrics/g_eval/templates/``.
    """
