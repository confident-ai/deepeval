from typing import List


class MistakeAvoidanceTemplate:
    """Prompt for mistake avoidance.

    A system that learns from failure keeps a record of what it got wrong
    before: a wrong answer it gave, the reasoning error behind it, a source
    it should not have trusted. This prompt checks, one known mistake at a
    time, whether the current answer makes that mistake again.
    """

    @staticmethod
    def number_mistakes(mistakes: List[str]) -> str:
        return "\n".join(
            f"{index + 1}. {mistake}" for index, mistake in enumerate(mistakes)
        )

    @staticmethod
    def generate_verdicts(
        input: str, actual_output: str, mistakes: List[str]
    ) -> str:
        return f"""You are checking whether an answer repeats mistakes that are already known.
You are given a question, a candidate answer, and a numbered list of known mistakes. Each known mistake describes something that was got wrong before on this or a similar question: a wrong answer, a faulty line of reasoning, or a wrong assumption.
For EACH known mistake, decide whether the candidate answer AVOIDS it.

Definitions:
- "yes": the candidate answer avoids the mistake. It does not state the wrong answer, does not rely on the faulty reasoning, and does not make the wrong assumption. An answer that the mistake does not apply to also counts as avoiding it.
- "no": the candidate answer repeats the mistake, in the same words or in different words.

Judge only against the known mistakes. Do not judge whether the answer is correct in any other way.

Return a JSON object with a single key "verdicts": a list with exactly {len(mistakes)} objects, one per known mistake and in the same order. Each object has two keys: "verdict" ("yes" or "no") and "reason" (one sentence). Do not return anything else.

Example known mistakes:
1. Answered "Sydney" because it is the largest city in Australia; the capital is Canberra.
2. Gave the population of the metropolitan area when the question asked about the city proper.

Example candidate answer:
The capital of Australia is Sydney.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "no",
            "reason": "The answer names Sydney as the capital, which is the known wrong answer."
        }},
        {{
            "verdict": "yes",
            "reason": "The answer gives no population figure, so this mistake does not apply."
        }}
    ]
}}
**

Question:
{input}

Known mistakes:
{MistakeAvoidanceTemplate.number_mistakes(mistakes)}

Candidate answer:
{actual_output}

JSON:
"""
