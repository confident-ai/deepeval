import pytest

from deepteam.vulnerabilities import IndirectInstruction
from deepteam.vulnerabilities.indirect_instruction import (
    IndirectInstructionType,
)
from deepteam.test_case import RTTestCase


class TestIndirectInstruction:

    def test_indirect_instruction_all_types(self):
        types = [
            "rag_injection",
            "tool_output_injection",
            "document_embedded_instructions",
            "cross_context_injection",
        ]
        indirect_instruction = IndirectInstruction(types=types)
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(types)

    def test_indirect_instruction_all_types_default(self):
        indirect_instruction = IndirectInstruction()
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(type.value for type in IndirectInstructionType)

    def test_indirect_instruction_rag_injection(self):
        types = ["rag_injection"]
        indirect_instruction = IndirectInstruction(types=types)
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(types)

    def test_indirect_instruction_tool_output_injection(self):
        types = ["tool_output_injection"]
        indirect_instruction = IndirectInstruction(types=types)
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(types)

    def test_indirect_instruction_document_embedded_instructions(self):
        types = ["document_embedded_instructions"]
        indirect_instruction = IndirectInstruction(types=types)
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(types)

    def test_indirect_instruction_cross_context_injection(self):
        types = ["cross_context_injection"]
        indirect_instruction = IndirectInstruction(types=types)
        assert sorted(
            type.value for type in indirect_instruction.types
        ) == sorted(types)

    def test_indirect_instruction_all_types_invalid(self):
        types = [
            "rag_injection",
            "tool_output_injection",
            "document_embedded_instructions",
            "cross_context_injection",
            "invalid",
        ]
        with pytest.raises(ValueError):
            IndirectInstruction(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        indirect_instruction = IndirectInstruction(types=["rag_injection"])
        test_cases = indirect_instruction.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Indirect Instruction" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == IndirectInstructionType.RAG_INJECTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        indirect_instruction = IndirectInstruction(
            types=["tool_output_injection"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = indirect_instruction.assess(
            model_callback=dummy_model_callback,
        )

        assert indirect_instruction.is_vulnerable() is not None
        assert (
            indirect_instruction.simulated_attacks is not None
            and isinstance(indirect_instruction.simulated_attacks, dict)
        )
        assert indirect_instruction.res is not None and isinstance(
            indirect_instruction.res, dict
        )
        assert IndirectInstructionType.TOOL_OUTPUT_INJECTION in results
        assert len(results[IndirectInstructionType.TOOL_OUTPUT_INJECTION]) == 1
        test_case = results[IndirectInstructionType.TOOL_OUTPUT_INJECTION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_indirect_instruction_metric(self):
        from deepteam.metrics import IndirectInstructionMetric

        indirect_instruction = IndirectInstruction(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = indirect_instruction._get_metric(
            IndirectInstructionType.TOOL_OUTPUT_INJECTION
        )
        assert isinstance(metric, IndirectInstructionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        indirect_instruction = IndirectInstruction(
            types=["cross_context_injection"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await indirect_instruction.a_assess(
            model_callback=dummy_model_callback,
        )

        assert indirect_instruction.is_vulnerable() is not None
        assert (
            indirect_instruction.simulated_attacks is not None
            and isinstance(indirect_instruction.simulated_attacks, dict)
        )
        assert indirect_instruction.res is not None and isinstance(
            indirect_instruction.res, dict
        )
        assert IndirectInstructionType.CROSS_CONTENT_INJECTION in results
        assert (
            len(results[IndirectInstructionType.CROSS_CONTENT_INJECTION]) == 1
        )
        test_case = results[IndirectInstructionType.CROSS_CONTENT_INJECTION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
