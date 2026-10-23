from typing import Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from deepeval.metrics.community.tool_argument_validation.schema import (
    ToolArgumentValidationResult,
    ToolCallValidationError,
)


class ToolArgumentValidationMetric(BaseMetric):
    """Deterministic validation of tool argument schema contracts.

    Validates that each ``ToolCall`` in ``test_case.tools_called`` conforms to its
    expected Pydantic schema contract defined in ``tool_schemas``.

    Unlike ``ToolCorrectnessMetric`` (which requires a pre-recorded golden truth
    in ``expected_tools``) and ``ArgumentCorrectnessMetric`` (which invokes an LLM judge
    to evaluate semantic intent), this metric performs fast, in-memory, deterministic
    type and constraint validation with zero LLM inference cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.TOOLS_CALLED,
    ]

    def __init__(
        self,
        tool_schemas: Dict[str, Type[BaseModel]],
        threshold: Optional[float] = 1.0,
        strict_mode: bool = False,
        allow_unregistered: bool = False,
        include_reason: bool = True,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if not isinstance(tool_schemas, dict) or len(tool_schemas) == 0:
            raise ValueError(
                "ToolArgumentValidationMetric requires a non-empty dictionary of "
                "`tool_schemas` mapping tool names to Pydantic BaseModel classes."
            )
        for tool_name, schema_cls in tool_schemas.items():
            if not (
                isinstance(schema_cls, type)
                and issubclass(schema_cls, BaseModel)
            ):
                raise TypeError(
                    f"Schema for tool '{tool_name}' must be a subclass of pydantic.BaseModel, "
                    f"got {type(schema_cls).__name__}."
                )

        self.tool_schemas = tool_schemas
        self.threshold = 1.0 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.allow_unregistered = allow_unregistered
        self.include_reason = include_reason
        self.verbose_mode = verbose_mode
        self.flaky = flaky

        # Deterministic metric: no LLM evaluation model required.
        self.model = None
        self.using_native_model = False
        self.async_mode = False
        self.validation_results: List[ToolArgumentValidationResult] = []

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self
        )
        self.test_case = test_case

        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            tools_called: List[ToolCall] = test_case.tools_called or []
            self.validation_results = []

            if len(tools_called) == 0:
                self.score = 1.0
                self.reason = "No tool calls provided"
                self.success = self.is_successful()
                if self.verbose_mode:
                    self.verbose_logs = construct_verbose_logs(
                        self,
                        steps=[
                            "Tools called: []",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )
                return self.score

            evaluated_results: List[ToolArgumentValidationResult] = []
            for idx, tool_call in enumerate(tools_called):
                result = self._validate_tool_call(tool_call, idx)
                if result is not None:
                    evaluated_results.append(result)
                    self.validation_results.append(result)

            total_evaluated = len(evaluated_results)
            if total_evaluated == 0:
                self.score = 1.0
                self.reason = "No registered tool calls to evaluate (all unregistered calls were ignored)."
                self.success = self.is_successful()
                if self.verbose_mode:
                    self.verbose_logs = construct_verbose_logs(
                        self,
                        steps=[
                            f"Tools called: {[t.name for t in tools_called]}",
                            "Evaluated calls: 0",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )
                return self.score

            valid_count = sum(1 for r in evaluated_results if r.is_valid)
            raw_score = valid_count / total_evaluated
            self.score = (
                0.0
                if self.strict_mode and raw_score < self.threshold
                else raw_score
            )
            self.success = self.is_successful()

            if self.include_reason:
                self.reason = self._generate_reason(
                    evaluated_results, valid_count, total_evaluated
                )
            else:
                self.reason = None

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Registered tool schemas: {list(self.tool_schemas.keys())}",
                        f"Tools called: {[t.name for t in tools_called]}",
                        f"Evaluated results:\n{self._format_diagnostics(evaluated_results)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """Asynchronous entrypoint; delegates directly to synchronous evaluation."""
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _validate_tool_call(
        self, tool_call: ToolCall, call_index: int
    ) -> Optional[ToolArgumentValidationResult]:
        tool_name = tool_call.name
        if tool_name not in self.tool_schemas:
            if self.allow_unregistered:
                return None
            return ToolArgumentValidationResult(
                tool_name=tool_name,
                call_index=call_index,
                is_valid=False,
                errors=[
                    ToolCallValidationError(
                        tool_name=tool_name,
                        call_index=call_index,
                        message=f"Tool '{tool_name}' is not registered in tool_schemas.",
                        error_type="unregistered_tool",
                        field_path=None,
                    )
                ],
            )

        schema = self.tool_schemas[tool_name]
        raw_params = tool_call.input_parameters
        errors: List[ToolCallValidationError] = []

        if raw_params is None:
            raw_params = {}

        if isinstance(raw_params, str):
            try:
                schema.model_validate_json(raw_params)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field = (
                        ".".join(str(loc) for loc in err.get("loc", [])) or None
                    )
                    errors.append(
                        ToolCallValidationError(
                            tool_name=tool_name,
                            call_index=call_index,
                            message=err.get("msg", "Invalid parameter"),
                            error_type=err.get("type", "validation_error"),
                            field_path=field,
                        )
                    )

        elif isinstance(raw_params, dict):
            try:
                schema.model_validate(raw_params)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field = (
                        ".".join(str(loc) for loc in err.get("loc", [])) or None
                    )
                    errors.append(
                        ToolCallValidationError(
                            tool_name=tool_name,
                            call_index=call_index,
                            message=err.get("msg", "Invalid parameter"),
                            error_type=err.get("type", "validation_error"),
                            field_path=field,
                        )
                    )
        else:
            errors.append(
                ToolCallValidationError(
                    tool_name=tool_name,
                    call_index=call_index,
                    message=f"input_parameters must be dict or JSON string, got {type(raw_params).__name__}.",
                    error_type="type_error",
                    field_path=None,
                )
            )

        return ToolArgumentValidationResult(
            tool_name=tool_name,
            call_index=call_index,
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _generate_reason(
        self,
        results: List[ToolArgumentValidationResult],
        valid_count: int,
        total_count: int,
    ) -> str:
        if valid_count == total_count:
            return f"All {total_count} tool call(s) conformed to their schema contracts."

        diagnostics = self._format_diagnostics(results)
        return (
            f"Validation Failed: {valid_count}/{total_count} tool call(s) adhered to "
            f"schema contracts (Score: {self.score:.2f}).\n"
            f"Diagnostics:\n{diagnostics}"
        )

    def _format_diagnostics(
        self, results: List[ToolArgumentValidationResult]
    ) -> str:
        lines = []
        for res in results:
            status = "SUCCESS" if res.is_valid else "FAILED"
            lines.append(
                f"- Call #{res.call_index} [{res.tool_name}]: {status}"
            )
            for err in res.errors:
                field_str = (
                    f"parameter '{err.field_path}': " if err.field_path else ""
                )
                lines.append(
                    f"  * {field_str}{err.message} [type={err.error_type}]"
                )
        return "\n".join(lines)

    @property
    def __name__(self):
        return "Tool Argument Validation"
