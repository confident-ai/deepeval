from typing import Optional, Union
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.g_eval.schema import *
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.utils import (
    copy_graph,
    is_valid_dag_from_roots,
    extract_required_params,
)


class DAGMetric(BaseMetric):

    def __init__(
        self,
        name: str,
        dag: DeepAcyclicGraph,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_dag_suffix: bool = True,
    ):
        if (
            is_valid_dag_from_roots(
                root_nodes=dag.root_nodes, multiturn=dag.multiturn
            )
            == False
        ):
            raise ValueError("Cycle detected in DAG graph.")

        self._verbose_steps: List[str] = []
        self.dag = copy_graph(dag)
        self.name = name
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_dag_suffix = _include_dag_suffix

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            extract_required_params(self.dag.root_nodes, self.dag.multiturn),
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        self._verbose_steps = []
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
                self.dag._execute(metric=self, test_case=test_case)
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        *self._verbose_steps,
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            extract_required_params(self.dag.root_nodes, self.dag.multiturn),
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        self._verbose_steps = []
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            await self.dag._a_execute(metric=self, test_case=test_case)
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    *self._verbose_steps,
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    def upload(self):
        from rich.console import Console
        from deepeval.confident.api import Api, Endpoints, HttpMethods
        from deepeval.metrics.dag.utils import construct_dag_upload_payload

        api = Api()
        payload = construct_dag_upload_payload(
            name=self.name, dag=self.dag, multi_turn=False
        )
        data, _ = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.METRICS_ENDPOINT,
            body=payload,
        )
        self.metric_id = data.get("id")
        if self.metric_id:
            Console().print(
                "[rgb(5,245,141)]✓[/rgb(5,245,141)] Metric "
                f"'{self.name}' [DAG] uploaded successfully "
                f"(id: [bold]{self.metric_id}[/bold])"
            )
        return data

    def pull(self):
        from rich.console import Console
        from deepeval.confident.api import Api, Endpoints, HttpMethods
        from deepeval.metrics.dag.utils import build_dag_from_payload

        api = Api()
        data, _ = api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.METRIC_ENDPOINT,
            url_params={"name": self.name},
        )
        dag_json = data.get("dag")
        if not dag_json:
            raise ValueError(
                f"Metric '{self.name}' has no DAG graph and cannot be pulled "
                "as a DAGMetric."
            )
        self.dag = build_dag_from_payload(
            dag_json, api=api, multiturn=False
        )
        self.metric_id = data.get("id")
        Console().print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Metric "
            f"'{self.name}' [DAG] pulled successfully"
        )
        return data

    @property
    def __name__(self):
        if self._include_dag_suffix:
            return f"{self.name} [DAG]"
        else:
            return self.name
