import asyncio
from enum import Enum
from typing import List, Optional, Tuple, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.test_case import RTTestCase
from deepteam.attacks.attack_simulator.utils import (
    generate_with_cost,
    a_generate_with_cost,
    add_cost,
)
from .template import AttackEngineTemplates
from .types import TransformedAttack, AttackVariations, ValidationResult


class AttackEngine:
    def __init__(
        self,
        simulator_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        variations: int = 1,
        generation_guidelines: Optional[List[str]] = None,
        purpose: Optional[str] = None,
    ):
        self.simulator_model, _ = initialize_model(simulator_model)
        self.variations = max(1, min(variations, 5))
        self.generation_guidelines = generation_guidelines or []
        self.purpose = purpose

    def refine(
        self,
        test_cases: List[RTTestCase],
        purpose: Optional[str] = None,
    ) -> List[RTTestCase]:
        refined_test_cases: List[RTTestCase] = []
        effective_purpose = purpose if purpose is not None else self.purpose

        for test_case in test_cases:
            base_input = test_case.input or ""
            vulnerability = test_case.vulnerability
            vulnerability_type = self._vulnerability_type_label(
                test_case.vulnerability_type
            )

            transform_prompt = AttackEngineTemplates.transform_attack_template(
                original_input=base_input,
                vulnerability=vulnerability,
                vulnerability_type=vulnerability_type,
                generation_guidelines=self.generation_guidelines,
            )
            transformed, refine_cost = generate_with_cost(
                transform_prompt, TransformedAttack, self.simulator_model
            )
            transformed_input = transformed.input.strip()
            if not transformed_input:
                transformed_input = base_input

            candidates = [transformed_input]
            if self.variations > 1:
                variation_prompt = (
                    AttackEngineTemplates.generate_variations_template(
                        transformed_input=transformed_input,
                        num_variations=self.variations,
                        vulnerability=vulnerability,
                        vulnerability_type=vulnerability_type,
                        generation_guidelines=self.generation_guidelines,
                    )
                )
                variations, variation_cost = generate_with_cost(
                    variation_prompt, AttackVariations, self.simulator_model
                )
                refine_cost = add_cost(refine_cost, variation_cost)
                candidates = [transformed_input] + [
                    item.strip() for item in variations.inputs
                ]

            valid_inputs, validation_cost = self._validate_candidates_with_llm(
                candidates=candidates,
                vulnerability=vulnerability,
                vulnerability_type=vulnerability_type,
                purpose=effective_purpose,
            )
            refine_cost = add_cost(refine_cost, validation_cost)
            if not valid_inputs:
                valid_inputs = [transformed_input]

            # Carry the baseline simulation_cost forward and add this refine
            # pass's cost, split evenly across the refined test cases produced.
            combined_cost = add_cost(test_case.simulation_cost, refine_cost)
            per_refined_cost = (
                combined_cost / len(valid_inputs)
                if combined_cost is not None and valid_inputs
                else combined_cost
            )
            refined_test_cases.extend(
                [
                    self._clone_with_new_input(
                        test_case, refined_input, per_refined_cost
                    )
                    for refined_input in valid_inputs
                ]
            )

        return refined_test_cases

    async def a_refine(
        self,
        test_cases: List[RTTestCase],
        purpose: Optional[str] = None,
    ) -> List[RTTestCase]:
        results = await asyncio.gather(
            *[
                self._a_refine_one(test_case, purpose=purpose)
                for test_case in test_cases
            ]
        )

        refined_test_cases: List[RTTestCase] = []
        for refined in results:
            refined_test_cases.extend(refined)
        return refined_test_cases

    async def _a_refine_one(
        self,
        test_case: RTTestCase,
        purpose: Optional[str] = None,
    ) -> List[RTTestCase]:
        effective_purpose = purpose if purpose is not None else self.purpose
        base_input = test_case.input or ""
        vulnerability = test_case.vulnerability
        vulnerability_type = self._vulnerability_type_label(
            test_case.vulnerability_type
        )

        transform_prompt = AttackEngineTemplates.transform_attack_template(
            original_input=base_input,
            vulnerability=vulnerability,
            vulnerability_type=vulnerability_type,
            generation_guidelines=self.generation_guidelines,
        )
        transformed, refine_cost = await a_generate_with_cost(
            transform_prompt, TransformedAttack, self.simulator_model
        )
        transformed_input = transformed.input.strip()
        if not transformed_input:
            transformed_input = base_input

        candidates = [transformed_input]
        if self.variations > 1:
            variation_prompt = (
                AttackEngineTemplates.generate_variations_template(
                    transformed_input=transformed_input,
                    num_variations=self.variations,
                    vulnerability=vulnerability,
                    vulnerability_type=vulnerability_type,
                    generation_guidelines=self.generation_guidelines,
                )
            )
            variations, variation_cost = await a_generate_with_cost(
                variation_prompt, AttackVariations, self.simulator_model
            )
            refine_cost = add_cost(refine_cost, variation_cost)
            candidates = [transformed_input] + [
                item.strip() for item in variations.inputs
            ]

        valid_inputs, validation_cost = (
            await self._a_validate_candidates_with_llm(
                candidates=candidates,
                vulnerability=vulnerability,
                vulnerability_type=vulnerability_type,
                purpose=effective_purpose,
            )
        )
        refine_cost = add_cost(refine_cost, validation_cost)
        if not valid_inputs:
            valid_inputs = [transformed_input]

        # Carry the baseline simulation_cost forward and add this refine pass's
        # cost, split evenly across the refined test cases produced.
        combined_cost = add_cost(test_case.simulation_cost, refine_cost)
        per_refined_cost = (
            combined_cost / len(valid_inputs)
            if combined_cost is not None and valid_inputs
            else combined_cost
        )
        return [
            self._clone_with_new_input(
                test_case, refined_input, per_refined_cost
            )
            for refined_input in valid_inputs
        ]

    @staticmethod
    def _vulnerability_type_label(
        vulnerability_type: Optional[Enum],
    ) -> Optional[str]:
        if vulnerability_type is None:
            return None
        if isinstance(vulnerability_type, Enum):
            return str(vulnerability_type.value)
        return str(vulnerability_type)

    def _validate_candidates_with_llm(
        self,
        candidates: List[str],
        vulnerability: str,
        vulnerability_type: Optional[str],
        purpose: Optional[str] = None,
    ) -> Tuple[List[str], Optional[float]]:
        effective_purpose = purpose if purpose is not None else self.purpose
        validated: List[str] = []
        validation_cost = None
        for candidate in candidates[: self.variations]:
            if not candidate or not candidate.strip():
                continue

            prompt = AttackEngineTemplates.validate_attack_template(
                candidate_input=candidate,
                vulnerability=vulnerability,
                vulnerability_type=vulnerability_type,
                purpose=effective_purpose,
            )
            try:
                result, cost = generate_with_cost(
                    prompt, ValidationResult, self.simulator_model
                )
                validation_cost = add_cost(validation_cost, cost)
                if result.is_valid:
                    validated.append(candidate)
            except Exception:
                continue

        return validated, validation_cost

    async def _a_validate_candidates_with_llm(
        self,
        candidates: List[str],
        vulnerability: str,
        vulnerability_type: Optional[str],
        purpose: Optional[str] = None,
    ) -> Tuple[List[str], Optional[float]]:
        effective_purpose = purpose if purpose is not None else self.purpose
        trimmed = [c for c in candidates[: self.variations] if c and c.strip()]
        if not trimmed:
            return []

        async def validate_one(candidate: str):
            prompt = AttackEngineTemplates.validate_attack_template(
                candidate_input=candidate,
                vulnerability=vulnerability,
                vulnerability_type=vulnerability_type,
                purpose=effective_purpose,
            )
            try:
                result, cost = await a_generate_with_cost(
                    prompt, ValidationResult, self.simulator_model
                )
                return (candidate if result.is_valid else None), cost
            except Exception:
                return None, None

        results = await asyncio.gather(*[validate_one(c) for c in trimmed])
        validated = [c for c, _ in results if c]
        validation_cost = None
        for _, cost in results:
            validation_cost = add_cost(validation_cost, cost)
        return validated, validation_cost

    @staticmethod
    def _clone_with_new_input(
        test_case: RTTestCase,
        refined_input: str,
        simulation_cost: Optional[float] = None,
    ) -> RTTestCase:
        return RTTestCase(
            vulnerability=test_case.vulnerability,
            vulnerability_type=test_case.vulnerability_type,
            input=refined_input,
            attack_method=test_case.attack_method,
            metadata=test_case.metadata,
            simulation_cost=simulation_cost,
        )
