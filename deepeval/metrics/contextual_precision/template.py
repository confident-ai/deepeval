from typing import List, Dict


class ContextualPrecisionTemplate:
    @staticmethod
    def generate_verdicts(
        input: str, expected_output: str, retrieval_context: List[str]
    ):
        document_count_str = f" ({len(retrieval_context)} document{'s' if len(retrieval_context) > 1 else ''})"
        return f"""Given the input, expected output, and retrieval context, please generate a list of JSON objects to determine whether each node in the retrieval context was remotely useful in arriving at the expected output.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. These JSON only contain the `verdict` key that outputs only 'yes' or 'no', and a `reason` key to justify the verdict. In your reason, you should aim to quote parts of the context.
Example Retrieval Context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect", "He won the Nobel Prize in 1968.", "There was a cat."]
Example Input: "Who won the Nobel Prize in 1968 and for what?"
Example Expected Output: "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "It clearly addresses the question by stating that 'Einstein won the Nobel Prize for his discovery of the photoelectric effect.'"
        }},
        {{
            "verdict": "yes",
            "reason": "The text verifies that the prize was indeed won in 1968."
        }},
        {{
            "verdict": "no",
            "reason": "'There was a cat' is not at all relevant to the topic of winning a Nobel Prize."
        }}
    ]  
}}
Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
**
        
Input:
{input}

Expected output:
{expected_output}

Retrieval Context{document_count_str}:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_reason(
        input: str, score: float, verdicts: List[Dict[str, str]]
    ):
        # given the input and retrieval context for this input, where the verdict is whether ... and the node is the ..., give a reason for the score
        return f"""Given the input, retrieval contexts, and contextual precision score, provide a CONCISE summary for the score. Explain why it is not higher, but also why it is at its current score.
The retrieval contexts is a list of JSON with three keys: `verdict`, `reason` (reason for the verdict) and `node`. `verdict` will be either 'yes' or 'no', which represents whether the corresponding 'node' in the retrieval context is relevant to the input. 
Contextual precision represents if the relevant nodes are ranked higher than irrelevant nodes. Also note that retrieval contexts is given IN THE ORDER OF THEIR RANKINGS.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_precision_score> because <your_reason>."
}}


DO NOT mention 'verdict' in your reason, but instead phrase it as irrelevant nodes. The term 'verdict' is just here for you to understand the broader scope of things.
Also DO NOT mention there are `reason` fields in the retrieval contexts you are presented with, instead just use the information in the `reason` field.
In your reason, you MUST USE the `reason`, QUOTES in the 'reason', and the node RANK (starting from 1, eg. first node) to explain why the 'no' verdicts should be ranked lower than the 'yes' verdicts.
When addressing nodes, make it explicit that they are nodes in retrieval contexts.
If the score is 1, keep it short and say something positive with an upbeat tone (but don't overdo it, otherwise it gets annoying).
**

Contextual Precision Score:
{score}

Input:
{input}

Retrieval Contexts:
{verdicts}

JSON:
"""
