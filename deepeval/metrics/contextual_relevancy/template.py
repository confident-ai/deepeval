from typing import List


class ContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(
        input: str,
        irrelevant_statements: List[str],
        relevant_statements: List[str],
        score: float,
    ):
        return f"""Based on the given input, reasons for why the retrieval context is irrelevant to the input, the statements in the retrieval context that is actually relevant to the retrieval context, and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
In your reason, you should quote data provided in the reasons for irrelevancy and relevant statements to support your point.

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_relevancy_score> because <your_reason>."
}}

If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**


Contextual Relevancy Score:
{score}

Input:
{input}

Reasons for why the retrieval context is irrelevant to the input:
{irrelevant_statements}

Statement in the retrieval context that is relevant to the input:
{relevant_statements}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""Based on the input and context, please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input. The JSON will be a list of 'verdicts', with 2 mandatory fields: 'verdict' and 'statement', and 1 optional field: 'reason'.
You should first extract statements found in the context, which are high level information found in the context, before deciding on a verdict and optionally a reason for each statement.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the statement is relevant to the input.
Provide a 'reason' ONLY IF verdict is no. You MUST quote the irrelevant parts of the statement to back up your reason.
If provided context contains no actual content or statements then: give \"no\" as a \"verdict\",
put context into \"statement\", and \"No statements found in provided context.\" into \"reason\".
**
IMPORTANT: Please make sure to only return in JSON format.
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "What were some of Einstein's achievements?"

Example:
{{
    "verdicts": [
        {{
            "statement": "Einstein won the Nobel Prize for his discovery of the photoelectric effect in 1968",
            "verdict": "yes"
        }},
        {{
            "statement": "There was a cat.",
            "reason": "The retrieval context contained the information 'There was a cat' when it has nothing to do with Einstein's achievements.",
            "verdict": "no"
        }}
    ]
}}
**

Input:
{input}

Context:
{context}

JSON:
"""
