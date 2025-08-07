from typing import Optional, List, Type, Union

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ToolCall,
    MCPMetaData,
    MCPToolCall,
    MCPResourceCall,
    MCPPromptCall
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.argument_correctness.template import (
    ArgumentCorrectnessTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.argument_correctness.schema import *


class ArgumentCorrectnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            ArgumentCorrectnessTemplate
        ] = ArgumentCorrectnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        
        if test_case.tools_called:
            self._required_params.append(LLMTestCaseParams.TOOLS_CALLED)
        if (test_case.mcp_tools_called is not None or
            test_case.mcp_resources_called is not None or
            test_case.mcp_prompts_called is not None):
            self._required_params.append(LLMTestCaseParams.MCP_DATA)

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
                    )
                )
            else:
                if test_case.tools_called and len(test_case.tools_called) == 0:
                    self.verdicts = []
                    self.tools_score = 1.0
                    self.tools_reason = "No tool calls provided"
                else:
                    self.verdicts: List[ArgumentCorrectnessVerdict] = (
                        self._generate_verdicts(
                            test_case.input, test_case.tools_called
                        )
                    )
                    self.tools_score = self._calculate_score()
                    self.tools_reason = self._generate_reason(test_case.input)
                if test_case.mcp_data:
                    self.mcp_data: List[MCPMetaData] = test_case.mcp_data
                    self.mcp_tools_called: Optional[List[MCPToolCall]] = test_case.mcp_tools_called
                    self.mcp_resources_called: Optional[List[MCPResourceCall]] = test_case.mcp_resources_called
                    self.mcp_prompts_called: Optional[List[MCPPromptCall]] = test_case.mcp_prompts_called
                    available_primitives, primitives_used = self._get_mcp_interaction_text()
                    self.mcp_interaction = available_primitives + "\n" + primitives_used
                    self.mcp_score: MCPArgsScore = self._get_mcp_args_used_score(test_case, available_primitives, primitives_used)

                self.score = self._calculate_final_score(test_case)
                self.reason = self._get_final_reason(test_case)
                self.success = self.score >= self.threshold
                steps = []
                if test_case.tools_called:
                    steps.append(f"Verdicts:\n{prettify_list(self.verdicts)}")
                    steps.append(f"Score : {self.tools_score}\nReason: {self.tools_reason}")
                if test_case.mcp_data:
                    steps.append(f"MCP Tools Score: {self.mcp_score.score}")
                    steps.append(f"MCP Tools Reason: {self.mcp_score.reason}")
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

        if test_case.tools_called:
            self._required_params.append(LLMTestCaseParams.TOOLS_CALLED)
            self._required_params.append(LLMTestCaseParams.EXPECTED_TOOLS)
        if (test_case.mcp_tools_called is not None or
            test_case.mcp_resources_called is not None or
            test_case.mcp_prompts_called is not None):
            self._required_params.append(LLMTestCaseParams.MCP_DATA)

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if test_case.tools_called and len(test_case.tools_called) == 0:
                self.verdicts = []
                self.score = 1.0
                self.reason = "No tool calls provided"
            else:
                self.verdicts: List[ArgumentCorrectnessVerdict] = (
                    await self._a_generate_verdicts(
                        test_case.input, test_case.tools_called
                    )
                )
                self.tools_score = self._calculate_score()
                self.tools_reason = await self._a_generate_reason(test_case.input)
                if test_case.mcp_data:
                    self.mcp_data: List[MCPMetaData] = test_case.mcp_data
                    self.mcp_tools_called: Optional[List[MCPToolCall]] = test_case.mcp_tools_called
                    self.mcp_resources_called: Optional[List[MCPResourceCall]] = test_case.mcp_resources_called
                    self.mcp_prompts_called: Optional[List[MCPPromptCall]] = test_case.mcp_prompts_called
                    available_primitives, primitives_used = self._get_mcp_interaction_text()
                    self.mcp_interaction = available_primitives + "\n" + primitives_used
                    self.mcp_score: MCPArgsScore = self._get_mcp_args_used_score(test_case, available_primitives, primitives_used)

            self.score = self._calculate_final_score(test_case)
            self.reason = self._get_final_reason(test_case)
            self.success = self.score >= self.threshold
            steps = []
            if test_case.tools_called:
                steps.append(f"Verdicts:\n{prettify_list(self.verdicts)}")
                steps.append(f"Score: {self.score}\nReason: {self.reason}")
            if test_case.mcp_data:
                steps.append(f"MCP Tools Score: {self.mcp_score.score}")
                steps.append(f"MCP Tools Reason: {self.mcp_score.reason}")
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=steps,
            )

            return self.score

    async def _a_generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.tools_score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ArgumentCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ArgumentCorrectnessScoreReason = (
                    await self.model.a_generate(
                        prompt=prompt, schema=ArgumentCorrectnessScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.tools_score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ArgumentCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ArgumentCorrectnessScoreReason = self.model.generate(
                    prompt=prompt, schema=ArgumentCorrectnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self,
        input: str,
        tools_called: List[ToolCall],
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            tools_called=tools_called,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                return [item for item in res.verdicts]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    ArgumentCorrectnessVerdict(**item)
                    for item in data["verdicts"]
                ]

    def _generate_verdicts(
        self,
        input: str,
        tools_called: List[ToolCall],
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            tools_called=tools_called,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    ArgumentCorrectnessVerdict(**item)
                    for item in data["verdicts"]
                ]
            
    def _get_mcp_args_used_score(self, test_case, available_primitives, primitives_used):
        prompt = ArgumentCorrectnessTemplate.get_mcp_argument_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=MCPArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPArgsScore = self.model.generate(prompt, schema=MCPArgsScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPArgsScore(**data)
        
    async def _a_get_mcp_args_used_score(self, test_case, available_primitives, primitives_used):
        prompt = ArgumentCorrectnessTemplate.get_mcp_argument_correctness_prompt(
            test_case, available_primitives, primitives_used
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=MCPArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: MCPArgsScore = await self.model.a_generate(prompt, schema=MCPArgsScore)
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return MCPArgsScore(**data)
    
    def _get_mcp_interaction_text(self) -> tuple[str, str]:
        for mcp_data in self.mcp_data:
            available_primitives = f"MCP Server {mcp_data.server_name}\n"
            available_primitives += (
                "\nAvailable Tools:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(tool), indent_level=4
                    )
                    for tool in mcp_data.available_tools
                )
                + "\n]"
            ) if mcp_data.available_tools else ""
            available_primitives += (
                "\nAvailable Resources:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(resource), indent_level=4
                    )
                    for resource in mcp_data.available_resources
                )
                + "\n]"
            ) if mcp_data.available_resources else ""
            available_primitives += (
                "\nAvailable Prompts:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(prompt), indent_level=4
                    )
                    for prompt in mcp_data.available_prompts
                )
                + "\n]"
            ) if mcp_data.available_prompts else ""
        primitives_used = "MCP Primitives Used: \n"
        primitives_used += (
            "\nMCP Tools Called:\n[\n"
            + ",\n".join(
                self.indent_multiline_string(
                    repr(mcp_tool_call), indent_level=4
                )
                for mcp_tool_call in self.mcp_tools_called
            )
            + "\n]"
        ) if self.mcp_tools_called else ""
        primitives_used += (
            "\nMCP Resources Called:\n[\n"
            + ",\n".join(
                self.indent_multiline_string(
                    repr(mcp_resource_call), indent_level=4
                )
                for mcp_resource_call in self.mcp_resources_called
            )
            + "\n]"
        ) if self.mcp_resources_called else ""
        primitives_used += (
            "\nMCP Prompts Called:\n[\n"
            + ",\n".join(
                self.indent_multiline_string(
                    repr(mcp_prompt_call), indent_level=4
                )
                for mcp_prompt_call in self.mcp_prompts_called
            )
            + "\n]"
        ) if self.mcp_prompts_called else ""

        return available_primitives, primitives_used
    
    def _calculate_final_score(self, test_case: LLMTestCase):
        if test_case.tools_called and test_case.mcp_data:
            return (self.tools_score + self.mcp_score.score) / 2
        elif test_case.mcp_data:
            return self.mcp_score.score
        else:
            return self.tools_score
        
    def _get_final_reason(self, test_case: LLMTestCase):
        if test_case.tools_called and test_case.mcp_data:
            reasons = (
                f"[\n"
                f"\t{self.tools_reason}, \n"
                f"\t{self.mcp_score.reason} \n"
                f"]"
            )
            return reasons
        elif test_case.mcp_data:
            return self.mcp_score.reason
        else:
            return self.tools_reason

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        correct_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                correct_count += 1

        score = correct_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

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
        return "Argument Correctness"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())