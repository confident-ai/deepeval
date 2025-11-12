"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from typing import Dict, Optional, List, Tuple, Union
from rich.progress import Progress

from deepeval.metrics import BaseArenaMetric
from deepeval.metrics.arena_g_eval.utils import format_arena_test_case
from deepeval.test_case import (
    LLMTestCaseParams,
    ArenaTestCase,
)
from deepeval.metrics.arena_g_eval.template import ArenaGEvalTemplate
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    check_arena_test_case_params,
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.arena_g_eval.schema import *
from deepeval.metrics.g_eval.utils import (
    construct_g_eval_params_string,
    validate_criteria_and_evaluation_steps,
    number_evaluation_steps,
)
from deepeval.utils import update_pbar


class ArenaGEval(BaseArenaMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
    ):
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        self.name = name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = (
            evaluation_steps
            if evaluation_steps and len(evaluation_steps) > 0
            else None
        )
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_g_eval_suffix = _include_g_eval_suffix

    def measure(
        self,
        test_case: ArenaTestCase,
        _show_indicator: bool = True,
        _progress: Optional[Progress] = None,
        _pbar_id: Optional[int] = None,
    ) -> str:
        check_arena_test_case_params(test_case, self.evaluation_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                    )
                )
            else:
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                if _progress:
                    update_pbar(_progress, _pbar_id)
                masked_winner, masked_reason, dummy_to_real_names = (
                    self._compare(test_case)
                )
                if _progress:
                    update_pbar(_progress, _pbar_id)
                self.winner = dummy_to_real_names[masked_winner]
                self.reason = self._generate_rewritten_reason(
                    masked_reason, dummy_to_real_names
                )
                if _progress:
                    update_pbar(_progress, _pbar_id)
                self.success = True
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Criteria:\n{self.criteria}",
                        f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                        f"Winner: {self.winner}",
                        f"Reason: {self.reason}",
                    ],
                )

            return self.winner

    async def a_measure(
        self,
        test_case: ArenaTestCase,
        _show_indicator: bool = True,
        _progress: Optional[Progress] = None,
        _pbar_id: Optional[int] = None,
    ) -> str:
        check_arena_test_case_params(test_case, self.evaluation_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.evaluation_steps: List[str] = (
                await self._a_generate_evaluation_steps()
            )
            if _progress:
                update_pbar(_progress, _pbar_id)
            masked_winner, masked_reason, dummy_to_real_names = (
                await self._a_compare(test_case)
            )
            if _progress:
                update_pbar(_progress, _pbar_id)
            self.winner = dummy_to_real_names[masked_winner]
            self.reason = await self._a_generate_rewritten_reason(
                masked_reason, dummy_to_real_names
            )
            if _progress:
                update_pbar(_progress, _pbar_id)
            self.success = True
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Criteria:\n{self.criteria}",
                    f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                    f"Winner: {self.winner}",
                    f"Reason: {self.reason}",
                ],
            )
            return self.winner

    async def _a_generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = ArenaGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:
            try:
                res: Steps = await self.model.a_generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    def _generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = ArenaGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:
            try:
                res: Steps = self.model.generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_compare(
        self,
        test_case: ArenaTestCase,
    ) -> Tuple[str, str, Dict[str, str]]:
        formatted_test_case, dummy_to_real_names = format_arena_test_case(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = ArenaGEvalTemplate.generate_arena_winner(
            evaluation_steps=number_evaluation_steps(self.evaluation_steps),
            test_case_contents=formatted_test_case,
            parameters=g_eval_params_str,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Winner)
            self.evaluation_cost += cost
            return res.winner, res.reason, dummy_to_real_names
        else:
            try:
                res: Winner = await self.model.a_generate(prompt, schema=Winner)
                return res.winner, res.reason, dummy_to_real_names
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["winner"], data["reason"], dummy_to_real_names

    def _compare(
        self,
        test_case: ArenaTestCase,
    ) -> Tuple[str, str, Dict[str, str]]:
        formatted_test_case, dummy_to_real_names = format_arena_test_case(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = ArenaGEvalTemplate.generate_arena_winner(
            evaluation_steps=number_evaluation_steps(self.evaluation_steps),
            test_case_contents=formatted_test_case,
            parameters=g_eval_params_str,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Winner)
            self.evaluation_cost += cost
            return res.winner, res.reason, dummy_to_real_names
        else:
            try:
                res: Winner = self.model.generate(prompt, schema=Winner)
                return res.winner, res.reason, dummy_to_real_names
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["winner"], data["reason"], dummy_to_real_names

    async def _a_generate_rewritten_reason(
        self,
        reason: str,
        dummy_to_real_names: Dict[str, str],
    ) -> str:
        prompt = ArenaGEvalTemplate.rewrite_reason(
            reason=reason,
            dummy_to_real_names=dummy_to_real_names,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=RewrittenReason
            )
            self.evaluation_cost += cost
            return res.rewritten_reason
        else:
            try:
                res: RewrittenReason = await self.model.a_generate(
                    prompt, schema=RewrittenReason
                )
                return res.rewritten_reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["rewritten_reason"]

    def _generate_rewritten_reason(
        self,
        reason: str,
        dummy_to_real_names: Dict[str, str],
    ) -> str:
        prompt = ArenaGEvalTemplate.rewrite_reason(
            reason=reason,
            dummy_to_real_names=dummy_to_real_names,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=RewrittenReason)
            self.evaluation_cost += cost
            return res.rewritten_reason
        else:
            try:
                res: RewrittenReason = self.model.generate(
                    prompt, schema=RewrittenReason
                )
                return res.rewritten_reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["rewritten_reason"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        if self._include_g_eval_suffix:
            return f"{self.name} [Arena GEval]"
        else:
            return self.name
