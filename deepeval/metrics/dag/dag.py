from typing import Optional, Union
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.g_eval.schema import *
from deepeval.metrics.dag.graph import BaseNode, DeepAcyclicGraph
from deepeval.metrics.dag.utils import is_valid_dag, extract_required_params


class DAGMetric(BaseMetric):
    def __init__(
        self,
        name: str,
        root_node: BaseNode,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_dag_suffix: bool = True,
    ):
        if is_valid_dag(root_node) == False:
            raise ValueError("Cycle detected in DAG graph.")

        self.root_node = root_node
        self.name = name
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_dag_suffix = _include_dag_suffix

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case, extract_required_params(self.root_node), self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                dag = DeepAcyclicGraph(
                    root_node=self.root_node, metric=self, test_case=test_case
                )
                self.score = dag._run()
                self.success = self.score >= self.threshold
                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case, extract_required_params(self.root_node), self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            dag = DeepAcyclicGraph(
                root_node=self.root_node, metric=self, test_case=test_case
            )
            self.score = await dag._a_run()
            self.success = self.score >= self.threshold
            return self.score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        if self._include_dag_suffix:
            return f"{self.name} (DAG)"
        else:
            return self.name
