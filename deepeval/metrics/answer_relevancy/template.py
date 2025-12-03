from typing import List
import textwrap


class AnswerRelevancyTemplate:
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Given the text, breakdown and generate a list of statements presented. Ambiguous statements and single words can be considered as statements, but only if outside of a coherent statement.

Example:
Example text: 
Our new laptop model features a high-resolution Retina display for crystal-clear visuals. It also includes a fast-charging battery, giving you up to 12 hours of usage on a single charge. For security, we’ve added fingerprint authentication and an encrypted SSD. Plus, every purchase comes with a one-year warranty and 24/7 customer support.

{{
    "statements": [
        "The new laptop model has a high-resolution Retina display.",
        "It includes a fast-charging battery with up to 12 hours of usage.",
        "Security features include fingerprint authentication and an encrypted SSD.",
        "Every purchase comes with a one-year warranty.",
        "24/7 customer support is included."
    ]
}}
===== END OF EXAMPLE ======
        
**
IMPORTANT: Please make sure to only return in valid and parseable JSON format, with the "statements" key mapping to a list of strings. No words or explanation are needed. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.
**

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str):
        return f"""For the provided list of statements, determine whether each statement is relevant to address the input.
Generate JSON objects with 'verdict' and 'reason' fields.
The 'verdict' should be 'yes' (relevant), 'no' (irrelevant), or 'idk' (ambiguous/supporting information).
Provide 'reason' ONLY for 'no' or 'idk' verdicts.
The statements are from an AI's actual output.

**
IMPORTANT: Please make sure to only return in valid and parseable JSON format, with the 'verdicts' key mapping to a list of JSON objects. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.

Expected JSON format:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "reason": <explanation_for_irrelevance>,
            "verdict": "no"
        }},
        {{
            "reason": <explanation_for_ambiguity>,
            "verdict": "idk"
        }}
    ]  
}}

Generate ONE verdict per statement - number of 'verdicts' MUST equal number of statements.
'verdict' must be STRICTLY 'yes', 'no', or 'idk':
- 'yes': statement is relevant to addressing the input
- 'no': statement is irrelevant to the input  
- 'idk': statement is ambiguous (not directly relevant but could be supporting information)
Provide 'reason' ONLY for 'no' or 'idk' verdicts.
**          

Input:
{input}

Statements:
{statements}

JSON:
"""

    @staticmethod
    def generate_reason(
        irrelevant_statements: List[str], input: str, score: float
    ):
        return f"""Given the answer relevancy score, the list of reasons of irrelevant statements made in the actual output, and the input, provide a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score.
The irrelevant statements represent things in the actual output that is irrelevant to addressing whatever is asked/talked about in the input.
If there is nothing irrelevant, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).


**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.

Example:
Example JSON:
{{
    "reason": "The score is <answer_relevancy_score> because <your_reason>."
}}
===== END OF EXAMPLE ======
**


Answer Relevancy Score:
{score}

Reasons why the score can't be higher based on irrelevant statements in the actual output:
{irrelevant_statements}

Input:
{input}

JSON:
"""
    
    @staticmethod
    def generate_multimodal_statements(actual_output: List[str]):

        return textwrap.dedent(
            f"""Given the text, breakdown and generate a list of statements presented. Ambiguous statements and single words can also be considered as statements.

            Example:
            Example text: Shoes. The shoes can be refunded at no extra cost. Thanks for asking the question!

            {{
                "statements": ["Shoes.", "Shoes can be refunded at no extra cost", "Thanks for asking the question!"]
            }}
            ===== END OF EXAMPLE ======
                    
            **
            IMPORTANT: Please make sure to only return in JSON format, with the "statements" key mapping to a list of strings. No words or explanation is needed.
            **

            Text:
            {actual_output}

            JSON:
            """
        )

    @staticmethod
    def generate_multimodal_verdicts(input, actual_output):
        return (
            [
                textwrap.dedent(
                    f"""For the provided list of statements (which can contain images), determine whether each statement or image is relevant to address the input.
                    Please generate a list of JSON with two keys: `verdict` and `reason`.
                    The 'verdict' key should STRICTLY be either a 'yes', 'idk' or 'no'. Answer 'yes' if the statement or image is relevant to addressing the original input, 'no' if the statement or image is irrelevant, and 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).
                    The 'reason' is the reason for the verdict.
                    Provide a 'reason' ONLY if the answer is 'no' or 'idk'. 
                    The provided statements are statements and images generated in the actual output.

                    **
                    IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key mapping to a list of JSON objects.
                    Example input: What should I do if there is an earthquake?
                    Example statements: ["Shoes.", "Thanks for asking the question!", "Is there anything else I can help you with?", "Duck and hide"]
                    Example JSON:
                    {{
                        "verdicts": [
                            {{
                                "reason": "The 'Shoes.' statement made in the actual output is completely irrelevant to the input, which asks about what to do in the event of an earthquake.",
                                "verdict": "no"
                            }},
                            {{
                                "reason": "The statement thanking the user for asking the question is not directly relevant to the input, but is not entirely irrelevant.",
                                "verdict": "idk"
                            }},
                            {{
                                "reason": "The question about whether there is anything else the user can help with is not directly relevant to the input, but is not entirely irrelevant.",
                                "verdict": "idk"
                            }},
                            {{
                                "verdict": "yes"
                            }}
                        ]  
                    }}

                    Since you are going to generate a verdict for each statement and image, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of `statements`.
                    **          

                    Input:
                    """
                )
            ]
            + input
            + [
                textwrap.dedent(
                    """
                    Statements:
                    """
                )
            ]
            + actual_output
            + [
                textwrap.dedent(
                    """
                    JSON:
                    """
                )
            ]
        )

    @staticmethod
    def generate_multimodal_reason(irrelevant_statements, input, score):
        return textwrap.dedent(
            f"""Given the answer relevancy score, the list of reasons of irrelevant statements made in the actual output, and the input, provide a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score.
            The irrelevant statements represent things in the actual output that is irrelevant to addressing whatever is asked/talked about in the input.
            If there is nothing irrelevant, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).


            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <answer_relevancy_score> because <your_reason>."
            }}
            **

            Answer Relevancy Score:
            {score}

            Reasons why the score can't be higher based on irrelevant statements in the actual output:
            {irrelevant_statements}

            Input:
            {input}

            JSON:
            """
        )

