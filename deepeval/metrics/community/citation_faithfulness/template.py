from typing import List


class CitationFaithfulnessTemplate:
    """Prompt for citation-attribution faithfulness.

    Unlike the FaithfulnessMetric (which checks whether a claim is supported by
    the retrieval context *somewhere*), this prompt checks attribution: whether
    each ``[N]`` citation marker in the answer points to a passage that actually
    supports the specific claim the marker is attached to. It is designed to
    catch misattribution, where a claim is cited to passage ``[A]`` that does not
    support it even though some other passage ``[B]`` in the context would.
    """

    @staticmethod
    def number_passages(retrieval_context: List[str]) -> str:
        return "\n\n".join(
            f"[{index + 1}] {passage}"
            for index, passage in enumerate(retrieval_context)
        )

    @staticmethod
    def generate_verdict(
        input: str, numbered_passages: str, actual_output: str
    ) -> str:
        return f"""You are a faithfulness judge for a retrieval-augmented question answering system.
You are given a question, a set of numbered retrieved passages, and a candidate answer that contains [N] citation markers (N is the passage number being cited).
Decide whether the candidate answer is FAITHFUL or UNFAITHFUL.
Definitions:
- FAITHFUL: every factual claim is supported by the passages, AND every citation marker [N] points to a passage that actually supports the specific claim it is attached to.
- UNFAITHFUL: at least one claim is unsupported, contradicts a passage, OR carries a citation [N] where passage N does NOT support that claim — even if some other passage would support it (this is citation misattribution).
Check each [N] marker: read passage N and confirm it supports the claim the marker is attached to. Judge relative to the passages only, not world knowledge.

Return a JSON object with two keys: "verdict" ("faithful" or "unfaithful") and "reasoning" (one sentence). Do not return anything else.

Example JSON:
{{
    "verdict": "unfaithful",
    "reasoning": "The completion-year claim is cited to passage [1], which only states the height; passage [2] supports it instead."
}}
**

Question:
{input}

Numbered passages:
{numbered_passages}

Candidate answer:
{actual_output}

JSON:
"""

    @staticmethod
    def number_passage_list(retrieval_context: List[str]) -> List[str]:
        return [
            f"[{index + 1}] {passage}"
            for index, passage in enumerate(retrieval_context)
        ]

    @staticmethod
    def _experimental_system_one_verdict() -> str:
        return (
            "Is `actual_output` faithful to `passages`? Answer yes only if "
            "every factual claim is supported by `passages` AND every [N] "
            "citation marker is attached to a claim that passage [N] itself "
            "supports. Answer no if any claim is unsupported or contradicts a "
            "passage, or if any [N] marker points to a passage that does not "
            "support its claim, even when another passage would. Judge "
            "relative to `passages` only, not world knowledge."
        )

    @staticmethod
    def _experimental_system_one_questions() -> str:
        return """[
  {
    "type": "noul",
    "statement": "Every [N] citation marker in `actual_output` is attached to a claim that passage [N] in `passages` supports.",
    "weight": 2
  },
  {
    "type": "noul",
    "statement": "Every factual claim in `actual_output` is supported by `passages`, and none contradicts them."
  },
  {
    "type": "score",
    "question": "How many of the [N] citation markers in `actual_output` point to a passage in `passages` that supports the claim they are attached to?",
    "levels": ["None of them", "Less than half", "Most of them", "All of them"]
  }
]"""
