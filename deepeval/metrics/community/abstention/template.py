from typing import List


class AbstentionTemplate:
    """Prompt for abstention (answer-vs-decline) correctness.

    The judge makes two independent determinations about a retrieval-augmented
    answer: (1) whether the retrieved passages actually contain enough
    information to answer the question, and (2) whether the candidate answer
    abstained (declined to answer / said the information was not available)
    rather than giving a substantive answer. The metric then checks that the
    decision matches: answer when the context supports one, abstain when it does
    not.
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
        return f"""You are an abstention judge for a retrieval-augmented question answering system.
You are given a question, a set of numbered retrieved passages, and a candidate answer.
Make two independent judgments and return them as booleans.

1. context_supports_answer: true if the retrieved passages contain enough information to answer the question; false if the answer is absent from them or the passages are irrelevant to the question. Judge relative to the passages only, not world knowledge.
2. output_abstained: true if the candidate answer declines to answer — for example it says it does not know, states the information is not in the context, or refuses — instead of giving a substantive answer. A substantive answer counts as not abstaining even if it is wrong.

Return a JSON object with exactly three keys: "context_supports_answer" (boolean), "output_abstained" (boolean), and "reasoning" (one sentence explaining both judgments). Do not return anything else.

Example JSON:
{{
    "context_supports_answer": false,
    "output_abstained": true,
    "reasoning": "The passages give the tower's height but never its daily visitor capacity, and the answer correctly says the context does not contain that figure."
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
