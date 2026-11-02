class AnswerConsistencyTemplate:
    """Prompt asking a judge how closely two independent responses to the same input agree.

    The rubric (``core_facts``, ``specifics``, ``supporting_detail``, ``stance``, ``format``,
    each scored 1-5) is defined in ``repeatability_evals.prompts.DEFAULT_DIMENSIONS`` so the
    metric, the library and its documentation stay in sync. The responses are inserted in a
    single pass inside tags, so text in a response cannot alter the prompt's structure.
    """

    @staticmethod
    def generate_verdict(input: str, response_a: str, response_b: str) -> str:
        from deepeval.metrics.community._consistency_utils import (
            require_repeatability_evals,
        )

        lib = require_repeatability_evals()
        return lib.render_prompt(
            lib.build_rubric_prompt(),
            query=input,
            item_a=response_a,
            item_b=response_b,
        )
