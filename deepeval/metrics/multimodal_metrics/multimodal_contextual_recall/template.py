from typing import Union, List
import textwrap

from deepeval.test_case import MLLMImage


class MultimodalContextualRecallTemplate:
    @staticmethod
    def generate_reason(
        expected_output, supportive_reasons, unsupportive_reasons, score
    ) -> List[Union[str, MLLMImage]]:
        return (
            [
                textwrap.dedent(
                    f"""Given the original expected output, a list of supportive reasons, and a list of unsupportive reasons (which is deduced directly from the 'expected output'), and a contextual recall score (closer to 1 the better), summarize a CONCISE reason for the score.
                    A supportive reason is the reason why a certain sentence or image in the original expected output can be attributed to the node in the retrieval context.
                    An unsupportive reason is the reason why a certain sentence or image in the original expected output cannot be attributed to anything in the retrieval context.
                    In your reason, you should related supportive/unsupportive reasons to the sentence or image number in expected output, and info regarding the node number in retrieval context to support your final reason. The first mention of "node(s)" should specify "node(s) in retrieval context)".

                    **
                    IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
                    Example JSON:
                    {{
                        "reason": "The score is <contextual_recall_score> because <your_reason>."
                    }}

                    DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
                    If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
                    **

                    Contextual Recall Score:
                    {score}

                    Expected Output:
                    """
                )
            ]
            + expected_output
            + [
                textwrap.dedent(
                    f"""Supportive Reasons:
                    {supportive_reasons}

                    Unsupportive Reasons:
                    {unsupportive_reasons}

                    JSON:"""
                )
            ]
        )

    @staticmethod
    def generate_verdicts(
        expected_output, retrieval_context
    ) -> List[Union[str, MLLMImage]]:
        return (
            [
                textwrap.dedent(
                    f"""For EACH sentence and image in the given expected output below, determine whether the sentence or image can be attributed to the nodes of retrieval contexts. Please generate a list of JSON with two keys: `verdict` and `reason`.
                    The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the sentence or image can be attributed to any parts of the retrieval context, else answer 'no'.
                    The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said sentence or image. A node is either a string or image, but not both (so do not group images and texts in the same nodes). You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 

                    **
                    IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects, each with two keys: `verdict` and `reason`.

                    {{
                        "verdicts": [
                            {{
                                "reason": "...",
                                "verdict": "yes"
                            }},
                            ...
                        ]  
                    }}

                    Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of sentences and images in the `expected output`.
                    **

                    Expected Output:
                    """
                )
            ]
            + expected_output
            + [
                textwrap.dedent(
                    """
                    Retrieval Context:
                    """
                )
            ]
            + MultimodalContextualRecallTemplate.id_retrieval_context(
                retrieval_context
            )
            + [
                textwrap.dedent(
                    """
                    JSON:
                    """
                )
            ]
        )

    @staticmethod
    def id_retrieval_context(retrieval_context) -> List[Union[str, MLLMImage]]:
        annotated_retrieval_context = []
        for i, context in enumerate(retrieval_context):
            if isinstance(context, str):
                annotated_retrieval_context.append(f"Node {i + 1}: {context}")
            elif isinstance(context, MLLMImage):
                annotated_retrieval_context.append(f"Node {i + 1}:")
                annotated_retrieval_context.append(context)
        return annotated_retrieval_context
