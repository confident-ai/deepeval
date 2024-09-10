class HallucinationTemplate:
    @staticmethod
    def generate_verdicts(actual_output, contexts):
        return f"""For each context in contexts, which is a list of strings, please generate a list of JSON objects to indicate whether the given 'actual output' agrees with EACH context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the given text agrees with the context. 
The 'reason' is the reason for the verdict. When the answer is 'no', try to provide a correction in the reason. 

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example contexts: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1968."]
Example actual output: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The actual output agrees with the provided context which states that Einstein won the Nobel Prize for his discovery of the photoelectric effect."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output contradicts the provided context which states that Einstein won the Nobel Prize in 1968, not 1969."
        }}
    ]  
}}

You should NOT incorporate any prior knowledge you have and take each context at face value. Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL TO {len(contexts)}.
You should FORGIVE cases where the actual output is lacking in detail, you should ONLY provide a 'no' answer if IT IS A CONTRADICTION.
**

Contexts:
{contexts}

Actual Output:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(factual_alignments, contradictions, score):
        return f"""Given a list of factual alignments and contradictions, which highlights alignment/contradictions between the `actual output` and `contexts, use it to provide a reason for the hallucination score in a CONCISELY. Note that The hallucination score ranges from 0 - 1, and the lower the better.

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <hallucination_score> because <your_reason>."
}}
**

Factual Alignments:
{factual_alignments}

Contradictions:
{contradictions}

Hallucination Score:
{score}

JSON:
"""
