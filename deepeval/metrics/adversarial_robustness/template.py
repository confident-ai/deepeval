import textwrap
from typing import List
from deepeval.metrics.adversarial_robustness.schema import PerturbationResult

class AdversarialRobustnessTemplate:
    @staticmethod
    def generate_reason(
        score: float,
        perturbation_type: str,
        results: List[PerturbationResult],
    ) -> str:
        
        successful_perturbations = [r for r in results if r.is_robust]
        failed_perturbations = [r for r in results if not r.is_robust]

        success_examples = ""
        if successful_perturbations:
            # Show up to 2 examples
            for r in successful_perturbations[:2]:
                success_examples += f"- Perturbed Input: '{r.perturbed_input}'\n  - Model Output: '{r.perturbed_output}' (Correctly matched original)\n"

        failure_examples = ""
        if failed_perturbations:
            # Show up to 2 examples
            for r in failed_perturbations[:2]:
                failure_examples += f"- Perturbed Input: '{r.perturbed_input}'\n  - Original Output: '{r.original_output}'\n  - Adversarial Output: '{r.perturbed_output}'\n"

        return textwrap.dedent(f"""
            Given the adversarial robustness score and examples of successful and failed perturbations, provide a concise reason for the score.

            **Analysis Summary**

            - **Adversarial Robustness Score**: {score:.2f}/1.00
            - **Perturbation Type**: {perturbation_type}
            - **Original Model Output**: "{results[0].original_output if results else 'N/A'}"

            **Robustness Analysis**:
            A score of {score:.2f} was calculated, indicating the model's level of resilience to {perturbation_type} perturbations.

            **Examples of Perturbations Where the Model Was Robust**:
            {success_examples if success_examples else "The model was not robust to any of the tested perturbations."}

            **Examples of Perturbations Where the Model Failed (Output Changed)**:
            {failure_examples if failure_examples else "The model was robust to all tested perturbations."}

            ---
            Based on this analysis, generate a concise, user-friendly reason that explains the robustness score.

            **
            IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
            Example JSON:
            {{
                "reason": "The score is {score:.2f} because the model remained consistent on most {perturbation_type} perturbations but failed on a few critical instances where the input was slightly altered, indicating some vulnerability."
            }}
            **

            JSON:
        """)