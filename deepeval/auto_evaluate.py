from typing import List, Dict, Callable, Union, Optional
from tqdm.asyncio import tqdm as atqdm
import asyncio
import json
import os

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from langchain_core.documents import Document

from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.integrations import captured_data, Frameworks, auto_eval_state
from deepeval.models.openai_embedding_model import OpenAIEmbeddingModel
from deepeval.dataset.utils import convert_goldens_to_test_cases
from deepeval.synthesizer import Synthesizer
from deepeval.tracing import trace_manager
from deepeval.metrics import BaseMetric
from deepeval.dataset import Golden
from deepeval import evaluate
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
)
from deepeval.models import DeepEvalBaseLLM


def auto_evaluate(
    target_model: Callable,
    metrics: List[BaseMetric],
    # Evaluate parameters
    hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    show_indicator: bool = True,
    print_results: bool = True,
    ignore_errors: bool = False,
    skip_on_missing_params: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    # Synthesizer parameters:
    synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    filtration_config: Optional[FiltrationConfig] = None,
    evolution_config: Optional[EvolutionConfig] = None,
    styling_config: Optional[StylingConfig] = None,
    include_expected_output: bool = True,
    max_goldens_per_context: int = 2,
    # Caching Parameters
    write_cache: bool = True,
    use_cache: bool = False,
    cache_dataset: bool = True,
    # Async parameters:
    synthesizer_async_mode: bool = True,
    evaluation_run_async: bool = True,
    throttle_value: int = 0,
    max_concurrent: int = 100,
):
    # make a dummy call to initialize the model
    async def make_dummy_call():
        await target_model("dummy")

    asyncio.run(make_dummy_call())

    tracer = get_active_tracer(auto_eval_state)

    # Retrieve captured data
    if tracer == Frameworks.LLAMAINDEX:
        index: BaseIndex = captured_data.get("base_index", None)
        query_engine: BaseQueryEngine = captured_data.get("query_engine", None)
        assert index or query_engine, (
            "The 'auto_evaluate' function requires at least one of 'QueryEngine' or 'Index' to be provided."
            "Please provide valid query engines and/or indeces in your LlamaIndex application."
        )
        prompts_dict = query_engine.get_prompts()
        doc_nodes = index._docstore.docs or captured_data.get("nodes", None)
        assert prompts_dict or doc_nodes, (
            "The 'auto_evaluate' function requires at least one of 'PromptTemplate' or 'Index' to be provided."
            "Please provide valid prompts and/or document indeces in your LlamaIndex application."
        )

    elif tracer == Frameworks.LANGCHAIN:
        doc_nodes: Document = captured_data.get("documents", None)
        assert doc_nodes, (
            "The 'auto_evaluate' function requires 'Documents' to be provided."
            "Please provide valid documents or texts in your LangChain application."
        )

    # Generate goldens
    raw_goldens = generate_goldens(
        doc_nodes=doc_nodes,
        include_expected_output=include_expected_output,
        max_goldens_per_context=max_goldens_per_context,
        max_concurrent=max_concurrent,
        cache_dataset=cache_dataset,
        async_mode=synthesizer_async_mode,
        synthesizer_model=synthesizer_model,
        filtration_config=filtration_config,
        evolution_config=evolution_config,
        styling_config=styling_config,
    )

    # Populate goldens by generating responses and extracting retrieval context
    populated_goldens = populate_goldens(
        target_model=target_model, goldens=raw_goldens, tracer=tracer
    )

    # Run Evaluate
    test_cases = convert_goldens_to_test_cases(populated_goldens)
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparameters,
        run_async=evaluation_run_async,
        show_indicator=show_indicator,
        print_results=print_results,
        write_cache=write_cache,
        use_cache=use_cache,
        ignore_errors=ignore_errors,
        skip_on_missing_params=skip_on_missing_params,
        verbose_mode=verbose_mode,
        identifier=identifier,
        throttle_value=throttle_value,
        max_concurrent=max_concurrent,
    )


################################################
# Gennerate Goldens ############################
################################################


def generate_goldens(
    doc_nodes: Dict,
    include_expected_output: bool,
    max_goldens_per_context: int,
    max_concurrent: int,
    cache_dataset: bool = True,
    async_mode: bool = True,
    filtration_config: Optional[FiltrationConfig] = None,
    evolution_config: Optional[EvolutionConfig] = None,
    styling_config: Optional[StylingConfig] = None,
    synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
):
    synthesizer_model
    cache_folder = "cache"
    os.makedirs(cache_folder, exist_ok=True)
    cache_file_path = os.path.join(cache_folder, "goldens.json")

    # Check if cache file exists
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as cache_file:
            goldens = [Golden(**golden) for golden in json.load(cache_file)]
    else:
        # Generate goldens and save to cache file
        if doc_nodes:
            goldens: List[Golden] = generate_goldens_from_nodes(
                doc_nodes=doc_nodes,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                max_concurrent=max_concurrent,
                async_mode=async_mode,
                filtration_config=filtration_config,
                evolution_config=evolution_config,
                styling_config=styling_config,
                synthesizer_model=styling_config,
            )
        else:
            # in the future generate goldens from prompts
            pass
        if cache_dataset:
            with open(cache_file_path, "w") as cache_file:
                json.dump(
                    [golden.model_dump() for golden in goldens],
                    cache_file,
                    indent=4,
                )
    return goldens


def generate_goldens_from_nodes(
    doc_nodes: Dict,
    include_expected_output: bool,
    max_goldens_per_context: int,
    max_concurrent: int,
    async_mode: bool = True,
    filtration_config: Optional[FiltrationConfig] = None,
    evolution_config: Optional[EvolutionConfig] = None,
    styling_config: Optional[StylingConfig] = None,
    synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
):
    # generate contexts from nodes
    context_generator = ContextGenerator(
        embedder=OpenAIEmbeddingModel(),
        _nodes=[node for _, node in doc_nodes.items()],
    )
    context_generator._load_nodes()
    contexts, _, _ = context_generator.generate_contexts_from_nodes(
        num_context=2
    )

    # generate and return goldens from docs
    synthesizer = Synthesizer(
        model=synthesizer_model,
        async_mode=async_mode,
        max_concurrent=max_concurrent,
        filtration_config=filtration_config,
        evolution_config=evolution_config,
        styling_config=styling_config,
    )
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=include_expected_output,
        max_goldens_per_context=max_goldens_per_context,
    )
    return goldens


################################################
# Gennerate Responses ##########################
################################################


def populate_goldens(
    target_model: Callable, goldens: List[Golden], tracer: Frameworks
):
    async def populate_golden(golden: Golden):
        output = await target_model(golden.input)
        golden.actual_output = output
        golden.retrieval_context = trace_manager.get_track_params().get(
            "retrieval_context"
        )
        return golden

    async def populate_goldens(goldens: List[Golden]):
        tasks = [populate_golden(golden) for golden in goldens]
        goldens = await atqdm.gather(
            *tasks, desc="Generating LLM Responses", total=len(tasks)
        )
        return goldens

    goldens = asyncio.run(populate_goldens(goldens=goldens))
    return goldens


################################################
# Get Active Tracer ############################
################################################


def get_active_tracer(auto_eval_state):
    for framework, is_active in auto_eval_state.items():
        if is_active:
            return framework
    return None
