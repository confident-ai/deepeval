class BasePromptTemplate:
    """Base class for a metric's customizable prompt template.

    Subclass the template class of a specific metric (e.g.
    ``AnswerRelevancyTemplate``) and override any of its documented static prompt
    methods to customize that metric's LLM-as-a-judge prompts, then pass your
    subclass to the metric via the ``evaluation_template`` argument::

        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

        class CustomTemplate(AnswerRelevancyTemplate):
            @staticmethod
            def generate_statements(actual_output):
                return f"...custom prompt using {actual_output}..."

        metric = AnswerRelevancyMetric(evaluation_template=CustomTemplate)

    Any prompt method you do not override falls back to deepeval's default
    template for that metric. Each overriding method receives the same keyword
    arguments the metric passes to its default template and must return the final
    prompt string. Accept ``**kwargs`` in your override if you want to stay
    forward-compatible with future template arguments.

    The overridable methods (and the arguments they receive) are listed in the
    docstring of each metric's template subclass; the default template bodies live
    under ``deepeval/metrics/<metric>/templates/``.
    """
