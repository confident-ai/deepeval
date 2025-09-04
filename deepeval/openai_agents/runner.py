from __future__ import annotations

from agents import Runner as BaseRunner, RunConfig, RunResult, RunResultStreaming
from deepeval.tracing.tracing import Observer
from deepeval.tracing.context import current_span_context


class Runner(BaseRunner):
    """
    A subclass of Runner that adds metric collection functionality.
    Extends all runner methods to support metric_collection parameter
    and automatically includes it in trace_metadata.
    """

    @classmethod
    async def run(cls, *args, **kwargs) -> RunResult:
        """
        Run a workflow starting at the given agent with metric collection support.
        Accepts all original Runner.run parameters plus optional metric_collection.
        """
        metric_collection = kwargs.pop('metric_collection', None)
        metrics = kwargs.pop('metrics', None)
        input = args[1] if len(args) >= 2 else kwargs.get('input', None)
        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics, 
            func_name="run",    
            function_kwargs={"input": input} # input is also set in the context
        ) as observer:
            current_span = current_span_context.get()
            if current_span:
                current_span.input = input # setting input in the context for representability
            res = await super().run(*args, **kwargs)
            observer.result = str(res)
        return res

    @classmethod
    def run_sync(cls, *args, **kwargs) -> RunResult:
        """
        Run a workflow synchronously with metric collection support.
        Accepts all original Runner.run_sync parameters plus optional metric_collection.
        """
        # Extract metric_collection from kwargs if present
        metric_collection = kwargs.pop('metric_collection', None)
        metrics = kwargs.pop('metrics', None)
        input = args[1] if len(args) >= 2 else kwargs.get('input', None)
        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics, 
            func_name="run_sync",
            function_kwargs={"input": input} # input is also set in the context
        ) as observer:
            current_span = current_span_context.get()
            if current_span:
                current_span.input = input # setting input in the context for representability
            res = super().run_sync(*args, **kwargs)
            observer.result = str(res)

        return res