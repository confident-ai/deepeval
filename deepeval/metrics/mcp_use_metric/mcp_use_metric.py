from typing import Optional, List, Union

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    MCPServer,
    MCPToolCall,
    MCPResourceCall,
    MCPPromptCall,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from .template import MCPUseMetricTemplate
from .schema import MCPPrimitivesScore, MCPArgsScore
from deepeval.metrics.api import metric_data_manager


class MCPUseMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.MCP_SERVERS,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
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
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                available_primitives, primitives_used = (
                    self._get_mcp_interaction_text(
                        mcp_servers=test_case.mcp_servers,
                        mcp_tools_called=test_case.mcp_tools_called or [],
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
                self.success = self.score >= self.threshold
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
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            available_primitives, primitives_used = (
                self._get_mcp_interaction_text(
                    mcp_servers=test_case.mcp_servers,
                    mcp_tools_called=test_case.mcp_tools_called or [],
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
            self.success = self.score >= self.threshold
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
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    def _get_primitives_used_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPPrimitivesScore:
        prompt = MCPUseMetricTemplate.get_primitive_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=MCPPrimitivesScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPPrimitivesScore = self.model.generate(
                    prompt, schema=MCPPrimitivesScore
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPPrimitivesScore(**data)

    async def _a_get_primitives_used_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPPrimitivesScore:
        prompt = MCPUseMetricTemplate.get_primitive_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=MCPPrimitivesScore
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPPrimitivesScore = await self.model.a_generate(
                    prompt, schema=MCPPrimitivesScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPPrimitivesScore(**data)

    def _get_argument_correctness_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPArgsScore:
        prompt = MCPUseMetricTemplate.get_mcp_argument_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=MCPArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPArgsScore = self.model.generate(
                    prompt, schema=MCPArgsScore
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPArgsScore(**data)

    async def _a_get_argument_correctness_score(
        self,
        test_case: LLMTestCase,
        available_primitives: str,
        primitives_used: str,
    ) -> MCPArgsScore:
        prompt = MCPUseMetricTemplate.get_mcp_argument_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=MCPArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPArgsScore = await self.model.a_generate(
                    prompt, schema=MCPArgsScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPArgsScore(**data)

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
    ) -> str:
        return (
            f"[\n"
            f"\t{primitives_used_score.reason}\n"
            f"\t{argument_correctness_score.reason}\n"
            f"]\n"
        )

    def _get_mcp_interaction_text(
        self,
        mcp_servers: List[MCPServer],
        mcp_tools_called: List[MCPToolCall],
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

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "MCP Use"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())
