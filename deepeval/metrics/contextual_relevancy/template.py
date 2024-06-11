class ContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(input, irrelevancies, score):
        return f"""Based on the given input, reasons for why the retrieval context is irrelevant to the input, and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
In your reason, you should quote data provided in the reasons for irrelevancy to support your point.

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
{irrelevancies}

JSON:
"""

    @staticmethod
    def generate_verdict(text, context):
        return f"""Based on the input and context, please generate a JSON object to indicate whether the context is relevant to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the context is relevant to the input. 
Provide a 'reason' ONLY IF verdict is no. You MUST quote the irrelevant parts of the context to back up your reason.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "When what was some of Einstein's achievements?"

Example:
{{
    "verdict": "no",
    "reason": "Although the context contains information about Einstein winning the Nobel Prize, it irrelevantly includes 'There was a cat' when it has nothing to do with Einstein's achievements."
}}
**

Input:
{text}

Context:
{context}

JSON:
"""
