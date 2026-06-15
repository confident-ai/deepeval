from typing import Optional, List
import textwrap


class TurnFaithfulnessTemplate:
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
    def generate_claims(
        input: str, assistant_output: str, multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""
            Extract every factual-sounding claim asserted in the ASSISTANT'S OUTPUT.

            A claim is any statement presented as fact, even if it is incorrect, vague, implied, or unverifiable.

            RULES:
            - Use ONLY the assistant's output as the source of claims.
            - Use the user's preceding message ONLY to resolve pronouns or references, not as factual evidence.
            - Extract claims exactly as stated without rewriting, summarizing, merging, or omitting details.
            - If a sentence contains multiple factual assertions, extract each as a separate claim.
            - Claims may involve text or images if multimodal.
            - Do NOT add, infer, or transform information.

            {TurnFaithfulnessTemplate.multimodal_rules if multimodal else ""}

            Output MUST be ONLY valid JSON:

            {{
                "claims": ["claim 1", "claim 2", ...]
            }}

            USER MESSAGE:
            {input}

            ASSISTANT OUTPUT:
            {assistant_output}

            JSON:
            """
        )

    @staticmethod
    def generate_truths(
        reference_context: str,
        extraction_limit: Optional[int],
        multimodal: bool = False,
    ):
        if extraction_limit is None:
            limit_description = "factual, explicit truths"
        elif extraction_limit == 1:
            limit_description = "one factual, explicit truth"
        else:
            limit_description = f"{extraction_limit} factual, explicit truths"

        return textwrap.dedent(
            f"""
            Extract {limit_description} from the REFERENCE CONTEXT.

            RULES:
            - Truths must be atomic, explicit factual statements.
            - Do not summarize or combine multiple facts.
            - Select truths based on reading order, not 'importance'.
            - Do not infer or expand beyond what is explicitly stated.
            - Keep each truth minimal but complete.
            - Treat images as factual evidence if multimodal, using only clearly visible information.

            {TurnFaithfulnessTemplate.multimodal_rules if multimodal else ""}

            Output MUST be ONLY valid JSON:

            {{
                "truths": ["truth 1", "truth 2", ...]
            }}

            REFERENCE CONTEXT:
            {reference_context}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(
        claims: List[str], reference_context: str, multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""
            For each claim, determine whether it is supported, contradicted, or not addressed by the reference context.

            DEFINITIONS:
            - "yes"  = The claim is directly supported by at least one truth.
            - "no"   = The claim directly contradicts at least one truth.
            - "idk"  = The context does not confirm or contradict the claim.

            RULES:
            - One verdict per claim, in the same order.
            - Do NOT use prior knowledge.
            - Only use the explicit truths provided.
            - A "yes" verdict must not include a reason.
            - A "no" or "idk" verdict must include a concise reason that quotes or paraphrases only the truths.
            - If a claim references an image and the visibility is unclear or ambiguous, use "idk".
            - Do not create new facts or explanations.

            {TurnFaithfulnessTemplate.multimodal_rules if multimodal else ""}

            Output MUST be ONLY valid JSON:

            {{
                "verdicts": [
                    {{
                        "verdict": "yes"
                    }},
                    {{
                        "verdict": "no",
                        "reason": "<explanation>"
                    }},
                    {{
                        "verdict": "idk",
                        "reason": "<explanation>"
                    }}
                ]
            }}

            REFERENCE CONTEXT:
            {reference_context}

            CLAIMS:
            {claims}

            JSON:
            """
        )

    @staticmethod
    def generate_reason(
        score: float, contradictions: List[str], multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""
            Below is a list of contradictions extracted from verdicts. Write a concise justification of the score.

            RULES:
            - If contradictions exist, summarize them in 1-3 sentences.
            - If no contradictions exist, respond:
              {{
                  "reason": "No contradictions were found."
              }}
            - The summary must reference only the contradictions listed.
            - Tone must be neutral and concise.
            - No external knowledge may be used.

            {TurnFaithfulnessTemplate.multimodal_rules if multimodal else ""}

            Output MUST be ONLY valid JSON:

            {{
                "reason": "<summary>"
            }}

            FAITHFULNESS SCORE:
            {score}

            CONTRADICTIONS:
            {contradictions}

            JSON:
            """
        )

    @staticmethod
    def generate_final_reason(
        final_score: float, success: bool, reasons: List[str]
    ):
        return textwrap.dedent(
            f"""You are an AI evaluator producing a single final explanation for the TurnFaithfulnessMetric result.

                Context:
                This metric evaluates conversational faithfulness by extracting truths from retrieval context, extracting claims from the assistant's output, and generating verdicts that compare each claim against the truths. Each interaction yields a reason indicating why a verdict failed or succeeded. You are given all those reasons.

                **
                IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
                Example JSON:
                {{
                    "reason": "The score is <turn_faithfulness_score> because <your_reason>."
                }}

                Inputs:
                - final_score: the averaged score across all interactions.
                - success: whether the metric passed or failed
                - reasons: a list of textual reasons generated from individual verdicts.

                Instructions:
                1. Read all reasons and synthesize them into one unified explanation.
                2. Describe patterns of claim-truth mismatches, contradictions, hallucinations, unsupported statements, or image-related errors if present.
                3. Do not repeat every reason; merge them into a concise, coherent narrative.
                5. If the metric failed, state the dominant failure modes. If it passed, state why the model's claims aligned with truths.
                6. Output a single paragraph with no lists, no bullets, no markup.

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
