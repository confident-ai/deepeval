class ContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(input, irrelevant_sentences, score):
        return f"""Based on the given input, irrelevant sentences (list of JSON), and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
Irrelevant Sentences will contain JSONs with two keys: `sentence` and `node`. `sentence` is the actual sentence itself, and `node` is the node number from the `retrieval context` which it was drawn from. Specify that nodes are from retrieval context the first time you mention it.
In your reason, you should use data in the irrelevant sentences to support your point.

Contextual Relevancy Score:
{score}

Input:
{input}

Irrelevant Sentences:
{irrelevant_sentences}

Example:
The score is <contextual_relevancy_score> because <your_reason>.

** 
IMPORTANT:
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**

Reason:
"""

    @staticmethod
    def generate_verdicts(text, context):
        return f"""Based on the input and context, please generate a list of JSON objects to indicate whether each given sentence in the context relevant to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'sentence'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the sentence is relevant to the text. 
Copy the sentence and supply the value to the 'sentence' key ONLY IF verdict is no.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "sentence": "There was a cat"
        }}
    ]  
}}
**

Input:
{text}

Context:
{context}

JSON:
"""