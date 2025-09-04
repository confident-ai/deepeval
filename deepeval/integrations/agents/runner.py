from __future__ import annotations

from typing import Any
from agents import Agent, Runner as BaseRunner, RunConfig, RunResult, RunResultStreaming
from agents.run import TContext, TResponseInputItem, DEFAULT_MAX_TURNS, RunHooks


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
        # Extract metric_collection from kwargs if present
        metric_collection = kwargs.pop('metric_collection', None)
        
        # Handle metric_collection by adding it to trace_metadata
        if metric_collection is not None:
            run_config = kwargs.get('run_config')
            if run_config is None:
                run_config = RunConfig()
                kwargs['run_config'] = run_config
            
            if run_config.trace_metadata is None:
                run_config.trace_metadata = {}
            
            run_config.trace_metadata["metric_collection"] = metric_collection

        # Call the parent run method
        return await super().run(*args, **kwargs)

    @classmethod
    def run_sync(cls, *args, **kwargs) -> RunResult:
        """
        Run a workflow synchronously with metric collection support.
        Accepts all original Runner.run_sync parameters plus optional metric_collection.
        """
        # Extract metric_collection from kwargs if present
        metric_collection = kwargs.pop('metric_collection', None)
        
        # Handle metric_collection by adding it to trace_metadata
        if metric_collection is not None:
            run_config = kwargs.get('run_config')
            if run_config is None:
                run_config = RunConfig()
                kwargs['run_config'] = run_config
            
            if run_config.trace_metadata is None:
                run_config.trace_metadata = {}
            
            run_config.trace_metadata["metric_collection"] = metric_collection

        # Call the parent run_sync method
        return super().run_sync(*args, **kwargs)

    @classmethod
    def run_streamed(cls, *args, **kwargs) -> RunResultStreaming:
        """
        Run a workflow starting at the given agent in streaming mode with metric collection support.
        Accepts all original Runner.run_streamed parameters plus optional metric_collection.
        """
        # Extract metric_collection from kwargs if present
        metric_collection = kwargs.pop('metric_collection', None)
        
        # Handle metric_collection by adding it to trace_metadata
        if metric_collection is not None:
            run_config = kwargs.get('run_config')
            if run_config is None:
                run_config = RunConfig()
                kwargs['run_config'] = run_config
            
            if run_config.trace_metadata is None:
                run_config.trace_metadata = {}
            
            run_config.trace_metadata["metric_collection"] = metric_collection

        # Call the parent run_streamed method
        return super().run_streamed(*args, **kwargs)
