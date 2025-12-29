from typing import List, Union
import textwrap
from deepeval.test_case import MLLMImage


class TurnContextualRecallTemplate:
    multimodal_rules = """
        --- MULTIMODAL INPUT RULES ---
        - Treat image content as factual evidence.
        - Only reference visual details that are explicitly and clearly visible.
        - Do not infer or guess objects, text, or details not visibly present.
        - If an image is unclear or ambiguous, mark uncertainty explicitly.
        - When evaluating claims, compare them to BOTH textual and visual evidence.
        - If the claim references something not clearly visible, respond with 'idk'.
    """

    @staticmethod
    def generate_reason(
        expected_outcome: str,
        supportive_reasons: str,
        unsupportive_reasons: str,
        score: float,
        multimodal: bool = False,
    ):
        content_type = "sentence or image" if multimodal else "sentence"

        return textwrap.dedent(
            f"""Given the original assistant output, a list of supportive reasons, and a list of unsupportive reasons ({'which is' if multimodal else 'which are'} deduced directly from the {'"assistant output"' if multimodal else 'original assistant output'}), and a contextual recall score (closer to 1 the better), summarize a CONCISE reason for the score.
            A supportive reason is the reason why a certain {content_type} in the original assistant output can be attributed to the node in the retrieval context.
            An unsupportive reason is the reason why a certain {content_type} in the original assistant output cannot be attributed to anything in the retrieval context.
            In your reason, you should {'related' if multimodal else 'relate'} supportive/unsupportive reasons to the {content_type} number in assistant output, and {'info' if multimodal else 'include info'} regarding the node number in retrieval context to support your final reason. The first mention of "node(s)" should specify "node(s) in retrieval context{')' if multimodal else ''}.

            {TurnContextualRecallTemplate.multimodal_rules if multimodal else ""}
            
            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <contextual_recall_score> because <your_reason>."
            }}

            DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
            If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it{',' if multimodal else ''} otherwise it gets annoying).
            **

            Contextual Recall Score:
            {score}

            Assistant Output:
            {expected_outcome}

            Supportive Reasons:
            {supportive_reasons}

            Unsupportive Reasons:
            {unsupportive_reasons}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(
        expected_outcome: str,
        retrieval_context: List[Union[str, MLLMImage]],
        multimodal: bool = False,
    ):
        content_type = "sentence and image" if multimodal else "sentence"
        content_type_plural = (
            "sentences and images" if multimodal else "sentences"
        )
        content_or = "sentence or image" if multimodal else "sentence"

        # For multimodal, we need to annotate the retrieval context with node IDs
        context_to_display = (
            TurnContextualRecallTemplate.id_retrieval_context(retrieval_context)
            if multimodal
            else retrieval_context
        )

        node_instruction = ""
        if multimodal:
            node_instruction = " A node is either a string or image, but not both (so do not group images and texts in the same nodes)."

        return textwrap.dedent(
            f"""For EACH {content_type} in the given assistant output below, determine whether the {content_or} can be attributed to the nodes of retrieval contexts. Please generate a list of JSON with two keys: `verdict` and `reason`.
            The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the {content_or} can be attributed to any parts of the retrieval context, else answer 'no'.
            The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said {content_or}.{node_instruction} You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 

            {TurnContextualRecallTemplate.multimodal_rules if multimodal else ""}
            
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

            Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of {content_type_plural} in {'the' if multimodal else '`assistant output`'}{' `assistant output`' if multimodal else ''}.
            **

            Assistant Output:
            {expected_outcome}

            Retrieval Context:
            {context_to_display}

            JSON:
            """
        )

    @staticmethod
    def generate_final_reason(
        final_score: float, success: bool, reasons: List[str]
    ):
        return textwrap.dedent(
            f"""You are an AI evaluator producing a single final explanation for the TurnContextualRecallMetric result.

            Context:
            This metric evaluates conversational contextual recall by determining whether sentences in the assistant output can be attributed to the retrieval context for each interaction. Each interaction yields a reason indicating which sentences were supported or unsupported. You are given all those reasons.

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <contextual_recall_score> because <your_reason>."
            }}

            Inputs:
            - final_score: the averaged score across all interactions.
            - success: whether the metric passed or failed
            - reasons: a list of textual reasons generated from individual interactions.

            Instructions:
            1. Read all reasons and synthesize them into one unified explanation.
            2. Describe patterns of unsupported sentences, missing context coverage, or well-attributed outputs if present.
            3. Do not repeat every reason; merge them into a concise, coherent narrative.
            4. If the metric failed, state the dominant failure modes. If it passed, state why the assistant output was well-supported by retrieval context.
            5. Output a single paragraph with no lists, no bullets, no markup.

            Output:
            A single paragraph explaining the final outcome.

            Here's the inputs:

            Final Score: {final_score}
            
            Reasons: 
            {reasons}

            Success: {success}

            Now give me a final reason that explains why the metric passed or failed. Output ONLY the reason and nothing else.

            JSON:
            """
        )

    @staticmethod
    def id_retrieval_context(
        retrieval_context: List[Union[str, MLLMImage]],
    ) -> List[Union[str, MLLMImage]]:
        """
        Annotates retrieval context with node IDs for multimodal processing.

        Args:
            retrieval_context: List of contexts (can be strings or MLLMImages)

        Returns:
            Annotated list with "Node X:" prefixes
        """
        annotated_retrieval_context = []
        for i, context in enumerate(retrieval_context):
            if isinstance(context, str):
                annotated_retrieval_context.append(f"Node {i + 1}: {context}")
            elif isinstance(context, MLLMImage):
                annotated_retrieval_context.append(f"Node {i + 1}:")
                annotated_retrieval_context.append(context)
        return annotated_retrieval_context
