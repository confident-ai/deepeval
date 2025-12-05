from typing import Optional, List
import textwrap


class FaithfulnessTemplate:
    @staticmethod
    def generate_claims(actual_output: str, multimodal: bool = False):
        multimodal_instruction = ""
        if multimodal:
            multimodal_instruction = " The excerpt may contain both text and images, so extract claims from all provided content."

        return textwrap.dedent(
            f"""Based on the given {'excerpt' if multimodal else 'text'}, please extract a comprehensive list of FACTUAL, undisputed truths, that can inferred from the provided actual AI output. {multimodal_instruction}
            These truths, MUST BE COHERENT, and CANNOT be taken out of context.
                
            Example:
            Example Text: 
            "Albert Einstein, the genius often associated with wild hair and mind-bending theories, famously won the Nobel Prize in Physics—though not for his groundbreaking work on relativity, as many assume. Instead, in 1968, he was honored for his discovery of the photoelectric effect, a phenomenon that laid the foundation for quantum mechanics."

            Example JSON: 
            {{
                "claims": [
                    "Einstein won the noble prize for his discovery of the photoelectric effect in 1968.",
                    "The photoelectric effect is a phenomenon that laid the foundation for quantum mechanics."
                ]  
            }}
            ===== END OF EXAMPLE ======

            **
            IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explanation is needed.
            Only include claims that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT. The claims you extract should include the full context it was presented in, NOT cherry picked facts.
            You should NOT include any prior knowledge, and take the text at face value when extracting claims.
            You should be aware that it is an AI that is outputting these claims.
            **

            {'Excerpt' if multimodal else 'AI Output'}:
            {actual_output}

            JSON:
            """
        )

    @staticmethod
    def generate_truths(
        retrieval_context: str,
        extraction_limit: Optional[int] = None,
        multimodal: bool = False,
    ):
        if extraction_limit is None:
            limit = " FACTUAL, undisputed truths"
        elif extraction_limit == 1:
            limit = " the single most important FACTUAL, undisputed truth"
        else:
            limit = f" the {extraction_limit} most important FACTUAL, undisputed truths per document"

        multimodal_instruction = ""
        if multimodal:
            multimodal_instruction = (
                " The excerpt may contain both text and images."
            )

        return textwrap.dedent(
            f"""Based on the given {'excerpt (text and images)' if multimodal else 'text'}, please generate a comprehensive list of{limit}, that can inferred from the provided {'excerpt' if multimodal else 'text'}.{multimodal_instruction}
            These truths, MUST BE COHERENT. They must NOT be taken out of context.
                    
            Example:
            Example Text: 
            "Albert Einstein, the genius often associated with wild hair and mind-bending theories, famously won the Nobel Prize in Physics—though not for his groundbreaking work on relativity, as many assume. Instead, in 1968, he was honored for his discovery of the photoelectric effect, a phenomenon that laid the foundation for quantum mechanics."

            Example JSON: 
            {{
                "truths": [
                    "Einstein won the noble prize for his discovery of the photoelectric effect in 1968.",
                    "The photoelectric effect is a phenomenon that laid the foundation for quantum mechanics."
                ]  
            }}
            ===== END OF EXAMPLE ======
            **
            IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explanation is needed.
            Only include truths that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT.
            **

            {'Excerpt' if multimodal else 'Text'}:
            {retrieval_context}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(
        claims: List[str], retrieval_context: str, multimodal: bool = False
    ):
        example_section = ""
        if multimodal:
            example_section = textwrap.dedent(
                """
                Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
                Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a German chef."]

                Example:
                {{
                    "verdicts": [
                        {{
                            "reason": "The claim about Barack Obama is not directly addressed in the retrieval context, and so poses no contradiction.",
                            "verdict": "idk"
                        }},
                        {{
                            "reason": "The claim about Zurich being a city in London is incorrect but does not pose a contradiction to the retrieval context.",
                            "verdict": "idk"
                        }},
                        {{
                            "verdict": "yes"
                        }},
                        {{
                            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead.",
                            "verdict": "no"
                        }},
                        {{
                            "reason": "The actual output claims Einstein is a German chef, which is not correct as the retrieval context states he was a German scientist instead.",
                            "verdict": "no"
                        }}
                    ]  
                }}
                ===== END OF EXAMPLE ======
                """
            )

        format_instruction = textwrap.dedent(
            """
            Expected JSON format:
            {{
                "verdicts": [
                    {{
                        "verdict": "yes"
                    }},
                    {{
                        "reason": <explanation_for_contradiction>,
                        "verdict": "no"
                    }},
                    {{
                        "reason": <explanation_for_uncertainty>,
                        "verdict": "idk"
                    }}
                ]  
            }}
            """
        )

        guidelines = ""
        if multimodal:
            guidelines = textwrap.dedent(
                """
                The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
                You DON'T have to provide a reason if the answer is 'yes'.
                ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
                Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
                Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
                If there are clear contradictions or any data or images that's not mentioned in the retrieval context, just provide 'no'.
                """
            )
        else:
            guidelines = textwrap.dedent(
                """
                Generate ONE verdict per claim - length of 'verdicts' MUST equal number of claims.
                No 'reason' needed for 'yes' verdicts.
                Only use 'no' if retrieval context DIRECTLY CONTRADICTS the claim - never use prior knowledge.
                Use 'idk' for claims not backed up by context OR factually incorrect but non-contradictory - do not assume your knowledge.
                Vague/speculative language in claims (e.g. 'may have', 'possibility') does NOT count as contradiction.
                """
            )

        return textwrap.dedent(
            f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
            The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
            Provide a 'reason' ONLY if the answer is 'no' or 'idk'. 
            The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

            {format_instruction}
            {example_section}
            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
            {guidelines}
            **

            Retrieval Contexts:
            {retrieval_context}

            Claims:
            {claims}

            JSON:
            """
        )

    @staticmethod
    def generate_reason(
        score: float, contradictions: List[str], multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
            Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the retrieval context (higher the better), CONCISELY summarize the contradictions to justify the score. 

            Expected JSON format:
            {{
                "reason": "The score is <faithfulness_score> because <your_reason>."
            }}

            ** 
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.

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
        )
