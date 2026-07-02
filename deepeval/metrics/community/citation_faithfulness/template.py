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
