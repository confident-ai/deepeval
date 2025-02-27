from typing import Optional, List


class FaithfulnessTemplate:
    @staticmethod
    def generate_claims(actual_output: str):
        return f"""Based on the given text, please extract a comprehensive list of FACTUAL, undisputed truths, that can inferred from the provided text.
These truths, MUST BE COHERENT, and CANNOT be taken out of context.
    
Example:
Example Text: 
"Albert Einstein, the genius often associated with wild hair and mind-bending theories, famously won the Nobel Prize in Physics—though not for his groundbreaking work on relativity, as many assume. Instead, in 1968, he was honored for his discovery of the photoelectric effect, a phenomenon that laid the foundation for quantum mechanics."

Example JSON: 
{{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect in 1968."
        "The photoelectric effect is a phenomenon that laid the foundation for quantum mechanics."
    ]  
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explanation is needed.
Only include claims that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT. The claims you extract should include the full context it was presented in, NOT cherry picked facts.
You should NOT include any prior knowledge, and take the text at face value when extracting claims.
**

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_truths(
        retrieval_context: str, extraction_limit: Optional[int] = None
    ):
        if extraction_limit is None:
            limit = " FACTUAL, undisputed truths"
        elif extraction_limit == 1:
            limit = " the single most important FACTUAL, undisputed truth"
        else:
            limit = f" the {extraction_limit} most important FACTUAL, undisputed truths per document"
        return f"""Based on the given text, please generate a comprehensive list of{limit}, that can inferred from the provided text.
These truths, MUST BE COHERENT. They must NOT be taken out of context.
        
Example:
Example Text: 
"Albert Einstein, the genius often associated with wild hair and mind-bending theories, famously won the Nobel Prize in Physics—though not for his groundbreaking work on relativity, as many assume. Instead, in 1968, he was honored for his discovery of the photoelectric effect, a phenomenon that laid the foundation for quantum mechanics."

Example JSON: 
{{
    "truths": [
        "Einstein won the noble prize for his discovery of the photoelectric effect in 1968."
        "The photoelectric effect is a phenomenon that laid the foundation for quantum mechanics."
    ]  
}}
===== END OF EXAMPLE ======
**
IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explanation is needed.
Only include truths that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT.
**

Text:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str):
        return f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
Provide a 'reason' ONLY if the answer is 'no'. 
The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }},
    ]  
}}
===== END OF EXAMPLE ======

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
**

Retrieval Contexts:
{retrieval_context}

Claims:
{claims}

JSON:
"""

    @staticmethod
    def generate_reason(score: float, contradictions: List[str]):
        return f"""Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the retrieval context (higher the better), CONCISELY summarize the contradictions to justify the score. 

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <faithfulness_score> because <your_reason>."
}}

If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `contradiction` in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Faithfulness Score:
{score}

Contradictions:
{contradictions}

JSON:
"""
