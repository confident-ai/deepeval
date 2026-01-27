from typing import List, Union
import textwrap
from deepeval.test_case import MLLMImage


class TurnContextualRelevancyTemplate:
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
        input: Union[str, List[Union[str, MLLMImage]]],
        irrelevant_statements: List[str],
        relevant_statements: List[str],
        score: float,
        multimodal: bool = False,
    ):
        # Note: irrelevancies parameter name in multimodal version is kept as irrelevant_statements for consistency
        return textwrap.dedent(
            f"""Based on the given user message, reasons for why the retrieval context is irrelevant to the user message, the statements in the retrieval context that is actually relevant to the retrieval context, and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
            In your reason, you should quote data provided in the reasons for irrelevancy and relevant statements to support your point.

            {TurnContextualRelevancyTemplate.multimodal_rules if multimodal else ""}
            
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

            User Message:
            {input}
            
            Reasons for why the retrieval context is irrelevant to the user message:
            {irrelevant_statements}

            Statement in the retrieval context that is relevant to the user message:
            {relevant_statements}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(
        input: Union[str, List[Union[str, MLLMImage]]],
        context: Union[str, List[Union[str, MLLMImage]]],
        multimodal: bool = False,
    ):
        context_type = "context (image or string)" if multimodal else "context"
        statement_or_image = "statement or image" if multimodal else "statement"

        # Conditional instructions based on mode
        extraction_instructions = ""
        if multimodal:
            extraction_instructions = textwrap.dedent(
                """
                If the context is textual, you should first extract the statements found in the context if the context, which are high level information found in the context, before deciding on a verdict and optionally a reason for each statement.
                If the context is an image, `statement` should be a description of the image. Do not assume any information not visibly available.
                """
            ).strip()
        else:
            extraction_instructions = "You should first extract statements found in the context, which are high level information found in the context, before deciding on a verdict and optionally a reason for each statement."

        # Additional instruction for empty context (only in non-multimodal)
        empty_context_instruction = ""
        if not multimodal:
            empty_context_instruction = '\nIf provided context contains no actual content or statements then: give "no" as a "verdict",\nput context into "statement", and "No statements found in provided context." into "reason".'

        return textwrap.dedent(
            f"""Based on the user message and {context_type}, please generate a JSON object to indicate whether {'the context' if multimodal else 'each statement found in the context'} is relevant to the provided user message. The JSON will be a list of 'verdicts', with 2 mandatory fields: 'verdict' and 'statement', and 1 optional field: 'reason'.
            {extraction_instructions}
            The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the {statement_or_image} is relevant to the user message.
            Provide a 'reason' ONLY IF verdict is no. You MUST quote the irrelevant parts of the {statement_or_image} to back up your reason.{empty_context_instruction}
            
            {TurnContextualRelevancyTemplate.multimodal_rules if multimodal else ""}

            **
            IMPORTANT: Please make sure to only return in JSON format.
            Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
            Example User Message: "What were some of Einstein's achievements?"

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

            User Message:
            {input}

            Context:
            {context}

            JSON:
            """
        )

    @staticmethod
    def generate_final_reason(
        final_score: float, success: bool, reasons: List[str]
    ):
        return textwrap.dedent(
            f"""You are an AI evaluator producing a single final explanation for the TurnContextualRelevancyMetric result.

            Context:
            This metric evaluates conversational contextual relevancy by determining whether statements in the retrieval context are relevant to the user message for each interaction. Each interaction yields a reason indicating which statements were relevant or irrelevant. You are given all those reasons.

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <contextual_relevancy_score> because <your_reason>."
            }}

            Inputs:
            - final_score: the averaged score across all interactions.
            - success: whether the metric passed or failed
            - reasons: a list of textual reasons generated from individual interactions.

            Instructions:
            1. Read all reasons and synthesize them into one unified explanation.
            2. Describe patterns of irrelevant statements, off-topic context, or well-targeted retrieval if present.
            3. Do not repeat every reason; merge them into a concise, coherent narrative.
            4. If the metric failed, state the dominant failure modes. If it passed, state why the retrieval context was relevant to user messages.
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
