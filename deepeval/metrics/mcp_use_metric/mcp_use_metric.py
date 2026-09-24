from typing import Any, Dict, Optional, List, Union, Type

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneEvalSpec,
    SystemOneScoreSpec,
    format_decision_reason,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_score,
    a_system_one_score,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    MCPServer,
    MCPToolCall,
    MCPResourceCall,
    MCPPromptCall,
    ToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.mcp.utils import mcp_calls_state, mcp_servers_state
from .schema import MCPPrimitivesScore, MCPArgsScore
from deepeval.templates import make_template_class


MCPUseTemplate = make_template_class("MCPUseMetric")

PRIMITIVE_USAGE_LEVELS = [
    "Wrong primitives",
    "Poor choice",
    "Reasonable choice",
    "Best choice",
]
ARGUMENT_CORRECTNESS_LEVELS = [
    "Incorrect",
    "Mostly incorrect",
    "Mostly correct",
    "Fully correct",
]


class MCPUseMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.MCP_SERVERS,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[MCPUseTemplate] = MCPUseTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                if run_system_one_eval(self, test_case):
                    return self.score

                available_primitives, primitives_used = (
                    self._get_mcp_interaction_text(
                        mcp_servers=test_case.mcp_servers,
                        mcp_tools_called=test_case.mcp_tools_called
                        or test_case.tools_called
                        or [],
                        mcp_resources_called=test_case.mcp_resources_called
                        or [],
                        mcp_prompts_called=test_case.mcp_prompts_called or [],
                    )
                )
                primitives_used_score = self._get_primitives_used_score(
                    test_case, available_primitives, primitives_used
                )
                argument_correctness_score = (
                    self._get_argument_correctness_score(
                        test_case, available_primitives, primitives_used
                    )
                )
                self.score = self._calculate_score(
                    primitives_used_score, argument_correctness_score
                )
                self.reason = self._get_reason(
                    primitives_used_score, argument_correctness_score
                )
                self.success = self.is_successful()
                steps = [
                    f"{available_primitives}",
                    f"{primitives_used}",
                    f"Primitive Usage Score: {primitives_used_score.score}",
                    f"Primitive Usage Reason: {primitives_used_score.reason}",
                    f"Argument Correctness Score: {argument_correctness_score.score}",
                    f"Argument Correctness Reason: {argument_correctness_score.reason}",
                ]
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=steps,
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            available_primitives, primitives_used = (
                self._get_mcp_interaction_text(
                    mcp_servers=test_case.mcp_servers,
                    mcp_tools_called=test_case.mcp_tools_called
                    or test_case.tools_called
                    or [],
                    mcp_resources_called=test_case.mcp_resources_called or [],
                    mcp_prompts_called=test_case.mcp_prompts_called or [],
                )
            )
            primitives_used_score = await self._a_get_primitives_used_score(
                test_case, available_primitives, primitives_used
            )
            argument_correctness_score = (
                await self._a_get_argument_correctness_score(
                    test_case, available_primitives, primitives_used
                )
            )
            self.score = self._calculate_score(
                primitives_used_score, argument_correctness_score
            )
            self.reason = self._get_reason(
                primitives_used_score, argument_correctness_score
            )
            self.success = self.is_successful()
            steps = [
                f"{available_primitives}",
                f"{primitives_used}",
                f"Primitive Usage Score: {primitives_used_score.score}",
                f"Primitive Usage Reason: {primitives_used_score.reason}",
                f"Argument Correctness Score: {argument_correctness_score.score}",
                f"Argument Correctness Reason: {argument_correctness_score.reason}",
            ]
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=steps,
            )
            return self.score

    def _get_primitives_used_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPPrimitivesScore:
        value = system_one_score(
            self, self._system_one_primitives_spec(test_case)
        )
        if value is not None:
            return MCPPrimitivesScore(
                score=value,
                reason=format_decision_reason(self, "primitive usage", value),
            )
        prompt = self._get_prompt(
            "get_primitive_correctness_prompt",
            template_class="MCPUseMetric",
            test_case=test_case,
            available_primitives=available_primitives,
            primitives_used=primitives_used,
            multimodal=test_case.multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MCPPrimitivesScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: MCPPrimitivesScore(**data),
        )

    async def _a_get_primitives_used_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPPrimitivesScore:
        value = await a_system_one_score(
            self, self._system_one_primitives_spec(test_case)
        )
        if value is not None:
            return MCPPrimitivesScore(
                score=value,
                reason=format_decision_reason(self, "primitive usage", value),
            )
        prompt = self._get_prompt(
            "get_primitive_correctness_prompt",
            template_class="MCPUseMetric",
            test_case=test_case,
            available_primitives=available_primitives,
            primitives_used=primitives_used,
            multimodal=test_case.multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MCPPrimitivesScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: MCPPrimitivesScore(**data),
        )

    def _get_argument_correctness_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPArgsScore:
        value = system_one_score(self, self._system_one_args_spec(test_case))
        if value is not None:
            return MCPArgsScore(
                score=value,
                reason=format_decision_reason(
                    self, "argument correctness", value
                ),
            )
        prompt = self._get_prompt(
            "get_mcp_argument_correctness_prompt",
            template_class="MCPUseMetric",
            test_case=test_case,
            available_primitives=available_primitives,
            primitives_used=primitives_used,
            multimodal=test_case.multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MCPArgsScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: MCPArgsScore(**data),
        )

    async def _a_get_argument_correctness_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPArgsScore:
        value = await a_system_one_score(
            self, self._system_one_args_spec(test_case)
        )
        if value is not None:
            return MCPArgsScore(
                score=value,
                reason=format_decision_reason(
                    self, "argument correctness", value
                ),
            )
        prompt = self._get_prompt(
            "get_mcp_argument_correctness_prompt",
            template_class="MCPUseMetric",
            test_case=test_case,
            available_primitives=available_primitives,
            primitives_used=primitives_used,
            multimodal=test_case.multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MCPArgsScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: MCPArgsScore(**data),
        )

    def _system_one_state(self, test_case: LLMTestCase) -> Dict[str, Any]:
        return {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "mcp_servers": mcp_servers_state(test_case.mcp_servers),
            "primitives_used": mcp_calls_state(
                test_case.mcp_tools_called or test_case.tools_called or [],
                test_case.mcp_resources_called or [],
                test_case.mcp_prompts_called or [],
            ),
        }

    def _system_one_primitives_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneScoreSpec]:
        if test_case.multimodal:
            return None
        return SystemOneScoreSpec(
            instructions=self._get_prompt(
                "_experimental_system_one_primitive_score"
            ),
            levels=PRIMITIVE_USAGE_LEVELS,
            state=self._system_one_state(test_case),
        )

    def _system_one_args_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneScoreSpec]:
        if test_case.multimodal:
            return None
        return SystemOneScoreSpec(
            instructions=self._get_prompt(
                "_experimental_system_one_args_score"
            ),
            levels=ARGUMENT_CORRECTNESS_LEVELS,
            state=self._system_one_state(test_case),
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `input`, `actual_output`, the MCP primitives called and the
        `mcp_servers` available; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.MCP_TOOLS_CALLED,
                SingleTurnParams.MCP_RESOURCES_CALLED,
                SingleTurnParams.MCP_PROMPTS_CALLED,
                SingleTurnParams.TOOLS_CALLED,
            ],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
            extra_state={
                "mcp_servers": mcp_servers_state(test_case.mcp_servers)
            },
        )

    def _calculate_score(
        self,
        primitives_used_score: MCPPrimitivesScore,
        argument_correctness_score: MCPArgsScore,
    ) -> float:
        score = min(
            primitives_used_score.score, argument_correctness_score.score
        )
        return 0 if self.strict_mode and score < self.threshold else score

    def _get_reason(
        self,
        primitives_used_score: MCPPrimitivesScore,
        argument_correctness_score: MCPArgsScore,
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        return (
            f"[\n"
            f"\t{primitives_used_score.reason}\n"
            f"\t{argument_correctness_score.reason}\n"
            f"]\n"
        )

    def _get_mcp_interaction_text(
        self,
        mcp_servers: List[MCPServer],
        mcp_tools_called: List[Union[MCPToolCall, ToolCall]],
        mcp_resources_called: List[MCPResourceCall],
        mcp_prompts_called: List[MCPPromptCall],
    ) -> tuple[str, str]:
        available_primitives = "MCP Primitives Available: \n"
        for mcp_server in mcp_servers:
            available_primitives += f"MCP Server {mcp_server.server_name}\n"
            available_primitives += (
                (
                    "\nAvailable Tools:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(repr(tool), indent_level=4)
                        for tool in mcp_server.available_tools
                    )
                    + "\n]"
                )
                if mcp_server.available_tools
                else ""
            )
            available_primitives += (
                (
                    "\nAvailable Resources:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(
                            repr(resource), indent_level=4
                        )
                        for resource in mcp_server.available_resources
                    )
                    + "\n]"
                )
                if mcp_server.available_resources
                else ""
            )
            available_primitives += (
                (
                    "\nAvailable Prompts:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(
                            repr(prompt), indent_level=4
                        )
                        for prompt in mcp_server.available_prompts
                    )
                    + "\n]"
                )
                if mcp_server.available_prompts
                else ""
            )
        primitives_used = "MCP Primitives Used: \n"
        primitives_used += (
            (
                "\nMCP Tools Called:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(mcp_tool_call), indent_level=4
                    )
                    for mcp_tool_call in mcp_tools_called
                )
                + "\n]"
            )
            if mcp_tools_called
            else ""
        )
        primitives_used += (
            (
                "\nMCP Resources Called:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(mcp_resource_call), indent_level=4
                    )
                    for mcp_resource_call in mcp_resources_called
                )
                + "\n]"
            )
            if mcp_resources_called
            else ""
        )
        primitives_used += (
            (
                "\nMCP Prompts Called:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(mcp_prompt_call), indent_level=4
                    )
                    for mcp_prompt_call in mcp_prompts_called
                )
                + "\n]"
            )
            if mcp_prompts_called
            else ""
        )

        return available_primitives, primitives_used

    @property
    def __name__(self):
        return "MCP Use"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())
