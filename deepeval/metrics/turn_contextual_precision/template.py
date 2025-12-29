from typing import List, Dict, Union
import textwrap
from deepeval.test_case import MLLMImage


class TurnContextualPrecisionTemplate:
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
    def generate_verdicts(
        input: str,
        expected_outcome: str,
        retrieval_context: List[str],
        multimodal: bool = False,
    ):
        document_count_str = f" ({len(retrieval_context)} document{'s' if len(retrieval_context) > 1 else ''})"

        # For multimodal, we need to annotate the retrieval context with node IDs
        context_to_display = (
            TurnContextualPrecisionTemplate.id_retrieval_context(
                retrieval_context
            )
            if multimodal
            else retrieval_context
        )

        multimodal_note = (
            " (which can be text or an image)" if multimodal else ""
        )

        prompt_template = textwrap.dedent(
            f"""Given the user message, assistant output, and retrieval context, please generate a list of JSON objects to determine whether each node in the retrieval context was remotely useful in arriving at the assistant output.

            {TurnContextualPrecisionTemplate.multimodal_rules if multimodal else ""}

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. These JSON only contain the `verdict` key that outputs only 'yes' or 'no', and a `reason` key to justify the verdict. In your reason, you should aim to quote parts of the context {multimodal_note}.
            Example Retrieval Context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect", "He won the Nobel Prize in 1968.", "There was a cat."]
            Example User Message: "Who won the Nobel Prize in 1968 and for what?"
            Example Assistant Output: "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect."

            Example:
            {{
                "verdicts": [
                    {{
                        "reason": "It clearly addresses the question by stating that 'Einstein won the Nobel Prize for his discovery of the photoelectric effect.'",
                        "verdict": "yes"
                    }},
                    {{
                        "reason": "The text verifies that the prize was indeed won in 1968.",
                        "verdict": "yes"
                    }},
                    {{
                        "reason": "'There was a cat' is not at all relevant to the topic of winning a Nobel Prize.",
                        "verdict": "no"
                    }}
                ]  
            }}
            Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
            **

            User Message:
            {input}

            Assistant Output:
            {expected_outcome}

            Retrieval Context {document_count_str}:
            {context_to_display}

            JSON:
            """
        )

        return prompt_template

    @staticmethod
    def generate_reason(
        input: str,
        score: float,
        verdicts: List[Dict[str, str]],
        multimodal: bool = False,
    ):
        return textwrap.dedent(
            f"""Given the user message, retrieval contexts, and contextual precision score, provide a CONCISE {'summarize' if multimodal else 'summary'} for the score. Explain why it is not higher, but also why it is at its current score.
            The retrieval contexts is a list of JSON with three keys: `verdict`, `reason` (reason for the verdict) and `node`. `verdict` will be either 'yes' or 'no', which represents whether the corresponding 'node' in the retrieval context is relevant to the user message. 
            Contextual precision represents if the relevant nodes are ranked higher than irrelevant nodes. Also note that retrieval contexts is given IN THE ORDER OF THEIR RANKINGS.

            {TurnContextualPrecisionTemplate.multimodal_rules if multimodal else ""}
            
            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <contextual_precision_score> because <your_reason>."
            }}


            DO NOT mention 'verdict' in your reason, but instead phrase it as irrelevant nodes. The term 'verdict' {'are' if multimodal else 'is'} just here for you to understand the broader scope of things.
            Also DO NOT mention there are `reason` fields in the retrieval contexts you are presented with, instead just use the information in the `reason` field.
            In your reason, you MUST USE the `reason`, QUOTES in the 'reason', and the node RANK (starting from 1, eg. first node) to explain why the 'no' verdicts should be ranked lower than the 'yes' verdicts.
            When addressing nodes, make it explicit that {'it is' if multimodal else 'they are'} nodes in {'retrieval context' if multimodal else 'retrieval contexts'}.
            If the score is 1, keep it short and say something positive with an upbeat tone (but don't overdo it{',' if multimodal else ''} otherwise it gets annoying).
            **

            Contextual Precision Score:
            {score}

            User Message:
            {input}

            Retrieval Contexts:
            {verdicts}

            JSON:
            """
        )

    @staticmethod
    def generate_final_reason(
        final_score: float, success: bool, reasons: List[str]
    ):
        return textwrap.dedent(
            f"""You are an AI evaluator producing a single final explanation for the TurnContextualPrecisionMetric result.

            Context:
            This metric evaluates conversational contextual precision by determining whether relevant nodes in retrieval context are ranked higher than irrelevant nodes for each interaction. Each interaction yields a reason indicating why relevant nodes were well-ranked or poorly-ranked. You are given all those reasons.

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is <contextual_precision_score> because <your_reason>."
            }}

            Inputs:
            - final_score: the averaged score across all interactions.
            - success: whether the metric passed or failed
            - reasons: a list of textual reasons generated from individual interactions.

            Instructions:
            1. Read all reasons and synthesize them into one unified explanation.
            2. Describe patterns of ranking issues, irrelevant nodes appearing before relevant ones, or well-structured retrieval contexts if present.
            3. Do not repeat every reason; merge them into a concise, coherent narrative.
            4. If the metric failed, state the dominant failure modes. If it passed, state why the retrieval context ranking was effective.
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
        retrieval_context: List[str],
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
