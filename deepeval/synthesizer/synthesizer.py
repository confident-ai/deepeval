from typing import List, Optional, Union, Tuple, Dict, Literal
from rich.progress import (
    Progress,
)
from rich.console import Console, Theme
from pydantic import BaseModel
from itertools import chain
import datetime
import asyncio
import random
import json
from rich import print
import tqdm
import csv
import os
from contextlib import nullcontext

from deepeval.utils import get_or_create_event_loop
from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.metrics.utils import (
    is_native_model,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.progress_context import synthesizer_progress_context
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset.golden import Golden
from deepeval.synthesizer.types import *
from deepeval.synthesizer.templates import (
    EvolutionTemplate,
    SynthesizerTemplate,
    FilterTemplate,
    PromptEvolutionTemplate,
    PromptSynthesizerTemplate,
    ExtractionTemplate,
)
from deepeval.synthesizer.schema import (
    SyntheticData,
    SyntheticDataList,
    SQLData,
    Response,
    InputFeedback,
    RewrittenInput,
    PromptStyling,
)
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from deepeval.synthesizer.utils import (
    print_synthesizer_status,
    SynthesizerStatus,
)
from deepeval.utils import update_pbar, add_pbar, remove_pbars

valid_file_types = ["csv", "json", "jsonl"]

evolution_map = {
    "Reasoning": EvolutionTemplate.reasoning_evolution,
    "Multi-context": EvolutionTemplate.multi_context_evolution,
    "Concretizing": EvolutionTemplate.concretizing_evolution,
    "Constrained": EvolutionTemplate.constrained_evolution,
    "Comparative": EvolutionTemplate.comparative_question_evolution,
    "Hypothetical": EvolutionTemplate.hypothetical_scenario_evolution,
    "In-Breadth": EvolutionTemplate.in_breadth_evolution,
}

prompt_evolution_map = {
    "Reasoning": PromptEvolutionTemplate.reasoning_evolution,
    "Concretizing": PromptEvolutionTemplate.concretizing_evolution,
    "Constrained": PromptEvolutionTemplate.constrained_evolution,
    "Comparative": PromptEvolutionTemplate.comparative_question_evolution,
    "Hypothetical": PromptEvolutionTemplate.hypothetical_scenario_evolution,
    "In-Breadth": PromptEvolutionTemplate.in_breadth_evolution,
}

my_theme = Theme({"progress.elapsed": "cyan"})
custom_console = Console(theme=my_theme)


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        max_concurrent: int = 100,
        filtration_config: Optional[FiltrationConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        styling_config: Optional[StylingConfig] = None,
        cost_tracking: bool = False,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.async_mode = async_mode
        self.max_concurrent = max_concurrent
        self.synthetic_goldens: List[Golden] = []
        self.filtration_config = (
            filtration_config
            if filtration_config is not None
            else FiltrationConfig(critic_model=self.model)
        )
        self.evolution_config = (
            evolution_config
            if evolution_config is not None
            else EvolutionConfig()
        )
        self.styling_config = (
            styling_config if styling_config is not None else StylingConfig()
        )
        self.set_styling_config = True if styling_config is not None else False
        self.cost_tracking = cost_tracking
        self.synthesis_cost = 0 if self.using_native_model else None

    #############################################################
    # Generate Goldens from Docs
    #############################################################

    def generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        context_construction_config: Optional[ContextConstructionConfig] = None,
        _send_data=True,
    ):
        self.synthetic_goldens = []
        self.synthesis_cost = 0 if self.using_native_model else None
        if context_construction_config is None:
            context_construction_config = ContextConstructionConfig(
                critic_model=self.model
            )

        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens = loop.run_until_complete(
                self.a_generate_goldens_from_docs(
                    document_paths=document_paths,
                    include_expected_output=include_expected_output,
                    max_goldens_per_context=max_goldens_per_context,
                    context_construction_config=context_construction_config,
                    _reset_cost=False,
                )
            )
        else:
            context_generator = ContextGenerator(
                document_paths=document_paths,
                encoding=context_construction_config.encoding,
                embedder=context_construction_config.embedder,
                chunk_size=context_construction_config.chunk_size,
                chunk_overlap=context_construction_config.chunk_overlap,
                model=context_construction_config.critic_model,
                filter_threshold=context_construction_config.context_quality_threshold,
                similarity_threshold=context_construction_config.context_similarity_threshold,
                max_retries=context_construction_config.max_retries,
            )
            num_contexts = (
                context_construction_config.max_contexts_per_document
                * len(document_paths)
            )
            total_goldens = num_contexts * max_goldens_per_context

            with synthesizer_progress_context(
                method="docs",
                evaluation_model=self.model.get_model_name(),
                num_evolutions=self.evolution_config.num_evolutions,
                evolutions=self.evolution_config.evolutions,
                embedder=context_construction_config.embedder.get_model_name(),
                max_generations=total_goldens,
                pbar_total=3 + num_contexts,
            ) as (progress, pbar_id), progress:

                # Generate contexts
                contexts, source_files, context_scores = (
                    context_generator.generate_contexts(
                        max_contexts_per_source_file=context_construction_config.max_contexts_per_document,
                        min_contexts_per_source_file=context_construction_config.min_contexts_per_document,
                        max_context_size=context_construction_config.max_context_length,
                        min_context_size=context_construction_config.min_context_length,
                        progress=progress,
                        pbar_id=pbar_id,
                    )
                )
                if self.synthesis_cost:
                    self.synthesis_cost += context_generator.total_cost
                print_synthesizer_status(
                    SynthesizerStatus.SUCCESS,
                    "Context Construction",
                    f"Utilizing {len(set(chain.from_iterable(contexts)))} out of {context_generator.total_chunks} chunks.",
                )
                advance = max(num_contexts - len(contexts), 0)
                (
                    update_pbar(progress, pbar_id, advance) if advance else None
                )  # prevent pbar removal error if advance is 0

                # Generate goldens from contexts
                goldens = self.generate_goldens_from_contexts(
                    contexts=contexts,
                    include_expected_output=include_expected_output,
                    max_goldens_per_context=max_goldens_per_context,
                    source_files=source_files,
                    _context_scores=context_scores,
                    _progress=progress,
                    _pbar_id=pbar_id,
                    _send_data=False,
                    _reset_cost=False,
                )
                if self.cost_tracking and self.using_native_model:
                    print(f"💰 API cost: {self.synthesis_cost:.6f}")
                if _send_data == True:
                    pass
                remove_pbars(
                    progress,
                    [
                        context_generator.pbar_generate_contexts_id,
                        context_generator.pbar_chunk_docs_id,
                        context_generator.pbar_load_docs_id,
                        pbar_id,
                    ],
                )

        return goldens

    async def a_generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        context_construction_config: Optional[ContextConstructionConfig] = None,
        _reset_cost=True,
    ):
        if context_construction_config is None:
            context_construction_config = ContextConstructionConfig(
                critic_model=self.model
            )
        if _reset_cost:
            self.synthesis_cost = 0 if self.using_native_model else None
            self.synthetic_goldens = []

        context_generator = ContextGenerator(
            document_paths=document_paths,
            encoding=context_construction_config.encoding,
            embedder=context_construction_config.embedder,
            chunk_size=context_construction_config.chunk_size,
            chunk_overlap=context_construction_config.chunk_overlap,
            model=context_construction_config.critic_model,
            filter_threshold=context_construction_config.context_quality_threshold,
            similarity_threshold=context_construction_config.context_similarity_threshold,
            max_retries=context_construction_config.max_retries,
        )
        num_contexts = (
            context_construction_config.max_contexts_per_document
            * len(document_paths)
        )
        total_goldens = num_contexts * max_goldens_per_context

        with synthesizer_progress_context(
            method="docs",
            evaluation_model=self.model.get_model_name(),
            num_evolutions=self.evolution_config.num_evolutions,
            evolutions=self.evolution_config.evolutions,
            embedder=context_construction_config.embedder.get_model_name(),
            max_generations=total_goldens,
            pbar_total=3 + num_contexts,
        ) as (progress, pbar_id), progress:

            # Generate contexts
            contexts, source_files, context_scores = (
                await context_generator.a_generate_contexts(
                    max_contexts_per_source_file=context_construction_config.max_contexts_per_document,
                    min_contexts_per_source_file=context_construction_config.min_contexts_per_document,
                    max_context_size=context_construction_config.max_context_length,
                    min_context_size=context_construction_config.min_context_length,
                    progress=progress,
                    pbar_id=pbar_id,
                )
            )
            if self.synthesis_cost:
                self.synthesis_cost += context_generator.total_cost
            print_synthesizer_status(
                SynthesizerStatus.SUCCESS,
                "Context Construction",
                f"Utilizing {len(set(chain.from_iterable(contexts)))} out of {context_generator.total_chunks} chunks.",
            )
            advance = max(num_contexts - len(contexts), 0)
            (
                update_pbar(progress, pbar_id, advance) if advance else None
            )  # prevent pbar removal error if advance is 0

            # Generate goldens from contexts
            goldens = await self.a_generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                source_files=source_files,
                _context_scores=context_scores,
                _progress=progress,
                _pbar_id=pbar_id,
                _reset_cost=False,
            )
            if _reset_cost and self.cost_tracking and self.using_native_model:
                print(f"💰 API cost: {self.synthesis_cost:.6f}")
            remove_pbars(
                progress,
                [
                    context_generator.pbar_generate_contexts_id,
                    context_generator.pbar_chunk_docs_id,
                    context_generator.pbar_load_docs_id,
                    pbar_id,
                ],
            )
            self.synthetic_goldens.extend(goldens)
            return goldens

    #############################################################
    # Generate Goldens from Contexts
    #############################################################

    def generate_goldens_from_contexts(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        source_files: Optional[List[str]] = None,
        _context_scores: Optional[List[float]] = None,
        _progress: Optional[Progress] = None,
        _pbar_id: Optional[int] = None,
        _send_data: bool = True,
        _reset_cost: bool = True,
    ) -> List[Golden]:
        if _reset_cost:
            self.synthetic_goldens = []
            self.synthesis_cost = 0 if self.using_native_model else None
        goldens: List[Golden] = []

        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens.extend(
                loop.run_until_complete(
                    self.a_generate_goldens_from_contexts(
                        contexts=contexts,
                        include_expected_output=include_expected_output,
                        max_goldens_per_context=max_goldens_per_context,
                        source_files=source_files,
                    )
                )
            )
        else:
            with synthesizer_progress_context(
                method="default",
                num_evolutions=self.evolution_config.num_evolutions,
                evolutions=self.evolution_config.evolutions,
                evaluation_model=self.model.get_model_name(),
                embedder=None,
                max_generations=len(contexts) * max_goldens_per_context,
                async_mode=False,
                progress=_progress,
                pbar_id=_pbar_id,
                pbar_total=len(contexts),
            ) as (progress, pbar_id), (
                progress if _progress is None else nullcontext()
            ):

                for context_index, context in enumerate(contexts):
                    # Calculate pbar lengths
                    should_style = (
                        self.styling_config.input_format
                        or self.styling_config.scenario
                        or self.styling_config.task
                    )
                    pbar_len_style = 1 if should_style else 0
                    pbar_len_expected_output = (
                        1 if include_expected_output else 0
                    )
                    pbar_len_evolve = (
                        self.evolution_config.num_evolutions
                        + pbar_len_style
                        + pbar_len_expected_output
                    )

                    # Add pbars
                    pbar_generate_goldens_id = add_pbar(
                        progress,
                        f"\t⚡ Generating goldens from context #{context_index}",
                        total=1 + max_goldens_per_context,
                    )
                    pbar_generate_inputs_id = add_pbar(
                        progress,
                        f"\t\t💡 Generating {max_goldens_per_context} input(s)",
                        total=2,
                    )
                    pbar_evolve_input_ids = []
                    for i in range(max_goldens_per_context):
                        pbar_evolve_input_ids.append(
                            add_pbar(
                                progress,
                                f"\t\t🧬 Evolving input #{i}",
                                total=pbar_len_evolve,
                            )
                        )

                    # Generate inputs
                    prompt = SynthesizerTemplate.generate_synthetic_inputs(
                        context=context,
                        max_goldens_per_context=max_goldens_per_context,
                        scenario=self.styling_config.scenario,
                        task=self.styling_config.task,
                        input_format=self.styling_config.input_format,
                    )
                    synthetic_inputs = self._generate_inputs(prompt)
                    update_pbar(progress, pbar_generate_inputs_id, remove=False)

                    # Qualify inputs
                    qualified_synthetic_inputs: List[SyntheticData]
                    scores: List[float]
                    qualified_synthetic_inputs, scores = self._rewrite_inputs(
                        context, synthetic_inputs
                    )
                    update_pbar(progress, pbar_generate_inputs_id, remove=False)
                    update_pbar(
                        progress, pbar_generate_goldens_id, remove=False
                    )

                    for input_index, data in enumerate(
                        qualified_synthetic_inputs
                    ):
                        # Evolve input
                        evolved_input, evolutions_used = self._evolve_input(
                            input=data.input,
                            context=context,
                            num_evolutions=self.evolution_config.num_evolutions,
                            evolutions=self.evolution_config.evolutions,
                            progress=progress,
                            pbar_evolve_input_id=pbar_evolve_input_ids[
                                input_index
                            ],
                            remove_pbar=False,
                        )

                        if should_style:
                            prompt = SynthesizerTemplate.rewrite_evolved_input(
                                input_format=self.styling_config.input_format,
                                evolved_input=evolved_input,
                                scenario=self.styling_config.scenario,
                                task=self.styling_config.task,
                            )
                            update_pbar(
                                progress,
                                pbar_evolve_input_ids[input_index],
                                remove=False,
                            )
                            res: SyntheticData = self._generate_schema(
                                prompt,
                                SyntheticData,
                                self.model,
                            )
                            evolved_input = res.input

                        # Synthesize Golden
                        golden = Golden(
                            input=evolved_input,
                            context=context,
                            source_file=(
                                source_files[context_index]
                                if source_files is not None
                                else None
                            ),
                            additional_metadata={
                                "evolutions": evolutions_used,
                                "synthetic_input_quality": scores[input_index],
                                "context_quality": (
                                    _context_scores[context_index]
                                    if _context_scores is not None
                                    else None
                                ),
                            },
                        )

                        # Generated expected output
                        if include_expected_output:
                            prompt = SynthesizerTemplate.generate_synthetic_expected_output(
                                input=golden.input,
                                context="\n".join(golden.context),
                                expected_output_format=self.styling_config.expected_output_format,
                            )
                            res = self._generate(prompt)
                            golden.expected_output = res
                            update_pbar(
                                progress,
                                pbar_evolve_input_ids[input_index],
                                remove=False,
                            )

                        goldens.append(golden)
                        update_pbar(
                            progress, pbar_generate_goldens_id, remove=False
                        )

                    # Add remaining progress if not enough goldens generated
                    update_pbar(progress, pbar_id, remove=False)
                    remove_pbars(
                        progress,
                        pbar_evolve_input_ids
                        + [pbar_generate_inputs_id, pbar_generate_goldens_id],
                    )

                # Remove pbar if not from docs
                remove_pbars(progress, [pbar_id]) if _progress is None else None

        if _send_data == True:
            pass
        if _reset_cost and self.cost_tracking and self.using_native_model:
            print(f"💰 API cost: {self.synthesis_cost:.6f}")
        self.synthetic_goldens.extend(goldens)
        return goldens

    async def a_generate_goldens_from_contexts(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        source_files: Optional[List[str]] = None,
        _context_scores: Optional[List[float]] = None,
        _progress: Optional[Progress] = None,
        _pbar_id: Optional[int] = None,
        _reset_cost: bool = True,
    ) -> List[Golden]:
        if _reset_cost:
            self.synthetic_goldens = []
            self.synthesis_cost = 0 if self.using_native_model else None
        semaphore = asyncio.Semaphore(self.max_concurrent)
        goldens: List[Golden] = []

        with synthesizer_progress_context(
            method="default",
            num_evolutions=self.evolution_config.num_evolutions,
            evolutions=self.evolution_config.evolutions,
            evaluation_model=self.model.get_model_name(),
            embedder=None,
            max_generations=len(contexts) * max_goldens_per_context,
            async_mode=True,
            pbar_id=_pbar_id,
            pbar_total=len(contexts),
            progress=_progress,
        ) as (progress, pbar_id), (
            progress if _progress is None else nullcontext()
        ):
            tasks = [
                self.task_wrapper(
                    semaphore,
                    self._a_generate_from_context,
                    semaphore=semaphore,
                    context=context,
                    goldens=goldens,
                    include_expected_output=include_expected_output,
                    max_goldens_per_context=max_goldens_per_context,
                    source_files=source_files,
                    index=index,
                    progress=progress,
                    pbar_id=pbar_id,
                    context_scores=_context_scores,
                )
                for index, context in enumerate(contexts)
            ]
            await asyncio.gather(*tasks)
            remove_pbars(progress, [pbar_id]) if _progress is None else None

        if _reset_cost and self.cost_tracking and self.using_native_model:
            print(f"💰 API cost: {self.synthesis_cost:.6f}")
        return goldens

    async def _a_generate_from_context(
        self,
        semaphore: asyncio.Semaphore,
        context: List[str],
        goldens: List[Golden],
        include_expected_output: bool,
        max_goldens_per_context: int,
        source_files: Optional[List[str]],
        index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        context_scores: Optional[List[float]] = None,
    ):
        # Calculate pbar lengths
        should_style = (
            self.styling_config.input_format
            or self.styling_config.scenario
            or self.styling_config.task
        )
        pbar_len_style = 1 if should_style else 0
        pbar_len_expected_output = 1 if include_expected_output else 0
        pbar_len_evolve = (
            self.evolution_config.num_evolutions
            + pbar_len_style
            + pbar_len_expected_output
        )

        # Add pbars
        pbar_generate_goldens_id = add_pbar(
            progress,
            f"\t⚡ Generating goldens from context #{index}",
            total=1 + max_goldens_per_context,
        )
        pbar_generate_inputs_id = add_pbar(
            progress,
            f"\t\t💡 Generating {max_goldens_per_context} input(s)",
            total=2,
        )
        pbar_evolve_input_ids = []
        for i in range(max_goldens_per_context):
            pbar_evolve_input_ids.append(
                add_pbar(
                    progress,
                    f"\t\t🧬 Evolving input #{i}",
                    total=pbar_len_evolve,
                )
            )

        # Generate inputs
        prompt = SynthesizerTemplate.generate_synthetic_inputs(
            context=context,
            max_goldens_per_context=max_goldens_per_context,
            scenario=self.styling_config.scenario,
            task=self.styling_config.task,
            input_format=self.styling_config.input_format,
        )
        synthetic_inputs: List[SyntheticData] = await self._a_generate_inputs(
            prompt
        )
        # Limit the length of the synthetic inputs to the maximum allowed
        synthetic_inputs = synthetic_inputs[:max_goldens_per_context]
        update_pbar(progress, pbar_generate_inputs_id, remove=False)

        # Qualify inputs
        qualified_synthetic_inputs: List[SyntheticData]
        scores: List[float]
        qualified_synthetic_inputs, scores = await self._a_rewrite_inputs(
            context, synthetic_inputs
        )
        update_pbar(progress, pbar_generate_inputs_id, remove=False)
        update_pbar(progress, pbar_generate_goldens_id, remove=False)

        # Helper function to process each input in parallel
        async def process_input(
            index: int,
            data: SyntheticData,
            progress: Optional[Progress] = None,
        ):
            # Evolve input
            evolved_input, evolutions_used = await self._a_evolve_input(
                input=data.input,
                context=context,
                num_evolutions=self.evolution_config.num_evolutions,
                evolutions=self.evolution_config.evolutions,
                progress=progress,
                pbar_evolve_input_id=pbar_evolve_input_ids[index],
                remove_pbar=False,
            )

            if should_style:
                prompt = SynthesizerTemplate.rewrite_evolved_input(
                    input_format=self.styling_config.input_format,
                    evolved_input=evolved_input,
                    scenario=self.styling_config.scenario,
                    task=self.styling_config.task,
                )
                res: SyntheticData = await self._a_generate_schema(
                    prompt,
                    SyntheticData,
                    self.model,
                )
                evolved_input = res.input
                update_pbar(
                    progress, pbar_evolve_input_ids[index], remove=False
                )

            # Generate expected output
            expected_output = None
            if include_expected_output:
                expected_output_prompt = SynthesizerTemplate.generate_synthetic_expected_output(
                    input=evolved_input,
                    context="\n".join(context),
                    expected_output_format=self.styling_config.expected_output_format,
                )
                expected_output = await self._a_generate(expected_output_prompt)
                update_pbar(
                    progress, pbar_evolve_input_ids[index], remove=False
                )

            # Create Golden
            golden = Golden(
                input=evolved_input,
                context=context,
                expected_output=expected_output,
                source_file=(
                    source_files[index]
                    if source_files is not None and index < len(source_files)
                    else None
                ),
                additional_metadata={
                    "evolutions": evolutions_used,
                    "synthetic_input_quality": scores[index],
                    # "context_quality": (
                    #     context_scores[data_index]
                    #     if context_scores is not None
                    #     else None
                    # ),
                },
            )
            update_pbar(progress, pbar_generate_goldens_id, remove=False)
            return golden

        # Process all inputs in parallel using asyncio.gather
        tasks = [
            self.task_wrapper(semaphore, process_input, index, data, progress)
            for index, data in enumerate(qualified_synthetic_inputs)
        ]
        results = await asyncio.gather(*tasks)

        # Add remaining progress if not enough goldens generated
        update_pbar(progress, pbar_id, remove=False)
        remove_pbars(
            progress,
            pbar_evolve_input_ids
            + [pbar_generate_inputs_id, pbar_generate_goldens_id],
        )
        goldens.extend(results)

    async def _a_generate_text_to_sql_from_context(
        self,
        context: List[str],
        goldens: List[Golden],
        include_expected_output: bool,
        max_goldens_per_context: int,
        progress_bar: tqdm.std.tqdm,
    ):
        # Generate inputs
        prompt = SynthesizerTemplate.generate_text2sql_inputs(
            context=context, max_goldens_per_context=max_goldens_per_context
        )
        synthetic_inputs: List[SyntheticData] = await self._a_generate_inputs(
            prompt
        )
        for data in synthetic_inputs:
            # Generate expected output
            expected_output = None
            if include_expected_output:
                prompt = SynthesizerTemplate.generate_text2sql_expected_output(
                    input=data.input, context="\n".join(context)
                )
                expected_output: SQLData = self._generate_schema(
                    prompt, SQLData, self.model
                )

            # Synthesize Golden
            golden = Golden(
                input=data.input,
                context=context,
                expected_output=expected_output.sql,
            )
            goldens.append(golden)

            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)

    #############################################################
    # Generate Goldens from Scratch
    #############################################################

    async def a_generate_goldens_from_scratch(
        self,
        num_goldens: int,
    ) -> List[Golden]:
        if (
            self.styling_config.scenario is None
            or self.styling_config.task is None
            or self.styling_config.input_format is None
        ):
            raise TypeError(
                "`scenario`, `task`, and `input_format` in `styling_config` must not be None when generation goldens from scratch."
            )
        self.synthetic_goldens = []
        self.synthesis_cost = 0 if self.using_native_model else None
        semaphore = asyncio.Semaphore(self.max_concurrent)

        transformed_evolutions = self.transform_distribution(
            self.evolution_config.evolutions
        )
        goldens: List[Golden] = []

        with synthesizer_progress_context(
            method="Scratch",
            num_evolutions=self.evolution_config.num_evolutions,
            evolutions=transformed_evolutions,
            evaluation_model=self.model.get_model_name(),
            embedder=None,
            max_generations=num_goldens,
            async_mode=True,
            pbar_total=num_goldens + 1,
        ) as (progress, pbar_id), progress:
            # Generate inputs
            prompt = PromptSynthesizerTemplate.generate_synthetic_prompts(
                scenario=self.styling_config.scenario,
                task=self.styling_config.task,
                input_format=self.styling_config.input_format,
                num_goldens=num_goldens,
            )
            synthetic_data = self._generate_inputs(prompt)
            update_pbar(progress, pbar_id)

            # Evolve inputs
            async def evolve_input(i, data: SyntheticData):
                pbar_evolve_input_id = add_pbar(
                    progress,
                    f"      🧬 Evolving inputs (#{i})",
                    total=self.evolution_config.num_evolutions,
                )
                evolved_prompts = await self.task_wrapper(
                    semaphore,
                    self._a_evolve_input,
                    input=data.input,
                    num_evolutions=self.evolution_config.num_evolutions,
                    evolutions=transformed_evolutions,
                    progress=progress,
                    pbar_evolve_input_id=pbar_evolve_input_id,
                )
                update_pbar(progress, pbar_id)
                return evolved_prompts

            tasks = [
                evolve_input(i, data) for i, data in enumerate(synthetic_data)
            ]
            evolved_prompts_list = await asyncio.gather(*tasks)

            # Synthesize Goldens
            goldens = [
                Golden(
                    input=evolved_prompt,
                    additional_metadata={"evolutions": evolutions},
                )
                for evolved_prompt, evolutions in evolved_prompts_list
            ]

        self.synthetic_goldens.extend(goldens)
        return goldens

    def generate_goldens_from_scratch(
        self,
        num_goldens: int,
        _send_data: bool = True,
    ) -> List[Golden]:
        if (
            self.styling_config.scenario is None
            or self.styling_config.task is None
            or self.styling_config.input_format is None
        ):
            raise TypeError(
                "`scenario`, `task`, and `input_format` in `styling_config` must not be None when generation goldens from scratch."
            )
        self.synthetic_goldens = []
        self.synthesis_cost = 0 if self.using_native_model else None

        transformed_evolutions = self.transform_distribution(
            self.evolution_config.evolutions
        )
        goldens: List[Golden] = []
        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens.extend(
                loop.run_until_complete(
                    self.a_generate_goldens_from_scratch(
                        num_goldens=num_goldens,
                    )
                )
            )
        else:
            with synthesizer_progress_context(
                method="Scratch",
                num_evolutions=self.evolution_config.num_evolutions,
                evolutions=transformed_evolutions,
                evaluation_model=self.model.get_model_name(),
                embedder=None,
                max_generations=num_goldens,
                async_mode=False,
                pbar_total=num_goldens + 1,
            ) as (progress, pbar_id), progress:

                # Generate inputs
                prompt = PromptSynthesizerTemplate.generate_synthetic_prompts(
                    scenario=self.styling_config.scenario,
                    task=self.styling_config.task,
                    input_format=self.styling_config.input_format,
                    num_goldens=num_goldens,
                )
                synthetic_data = self._generate_inputs(prompt)
                update_pbar(progress, pbar_id)

                # Evolve inputs
                for i, data in enumerate(synthetic_data):
                    pbar_evolve_input_id = add_pbar(
                        progress,
                        f"      🧬 Evolving inputs (#{i})",
                        total=self.evolution_config.num_evolutions,
                    )
                    evolved_prompt, evolutions_used = self._evolve_input(
                        input=data.input,
                        num_evolutions=self.evolution_config.num_evolutions,
                        evolutions=transformed_evolutions,
                        progress=progress,
                        pbar_evolve_input_id=pbar_evolve_input_id,
                    )
                    update_pbar(progress, pbar_id)

                # Synthesize Goldens
                golden = Golden(
                    input=evolved_prompt,
                    additional_metadata={"evolutions": evolutions_used},
                )
                goldens.append(golden)

        # Wrap up Synthesis
        self.synthetic_goldens.extend(goldens)
        if _send_data == True:
            pass
        return goldens

    def transform_distribution(
        self, evolutions: Dict[Evolution, float]
    ) -> Dict[PromptEvolution, float]:
        prompt_evolutions: Dict[PromptEvolution, float] = {}
        for evo, weight in evolutions.items():
            if evo == Evolution.MULTICONTEXT:
                continue
            prompt_evolution = self.map_evolution_to_prompt_evolution(evo)
            prompt_evolutions[prompt_evolution] = weight
        return prompt_evolutions

    def map_evolution_to_prompt_evolution(
        self, evolution: Evolution
    ) -> PromptEvolution:
        try:
            return PromptEvolution[evolution.name]
        except KeyError:
            raise KeyError(
                f"Evolution '{evolution.name}' not available for this method."
            )

    #############################################################
    # Generate from goldens
    #############################################################

    def generate_goldens_from_goldens(
        self,
        goldens: List[Golden],
        max_goldens_per_golden: int = 2,
        include_expected_output: bool = True,
    ) -> List[Golden]:
        self.synthetic_goldens = []
        if self.async_mode:
            loop = get_or_create_event_loop()
            result = loop.run_until_complete(
                self.a_generate_goldens_from_goldens(
                    goldens=goldens,
                    max_goldens_per_golden=max_goldens_per_golden,
                    include_expected_output=include_expected_output,
                )
            )
            self.synthetic_goldens.extend(result)
            return result
        else:
            # Extract contexts and source files from goldens
            contexts = []
            source_files = []
            for golden in goldens:
                if golden.context is None:
                    continue
                contexts.append(golden.context)
                source_files.append(golden.source_file)

            # Extract styles from goldens if not already set
            if self.set_styling_config == False:
                example_inputs = random.sample(
                    [golden.input for golden in goldens], min(len(goldens), 10)
                )
                styling_prompt = (
                    ExtractionTemplate.extract_prompt_structure_from_inputs(
                        example_inputs
                    )
                )
                styles = self._generate_schema(
                    styling_prompt, PromptStyling, self.model
                )
                styles_json = json.loads(styles.model_dump_json())
                styling_config = StylingConfig(
                    **styles_json, expected_output_format=None
                )
                self.styling_config = styling_config
            # Generate goldens from scratch or from contexts if available
            if len(contexts) == 0:
                return self.generate_goldens_from_scratch(
                    num_goldens=len(goldens) * max_goldens_per_golden,
                )
            else:
                return self.generate_goldens_from_contexts(
                    contexts=contexts,
                    include_expected_output=include_expected_output,
                    max_goldens_per_context=max_goldens_per_golden,
                    source_files=source_files,
                )

    async def a_generate_goldens_from_goldens(
        self,
        goldens: List[Golden],
        max_goldens_per_golden: int = 2,
        include_expected_output: bool = True,
    ) -> List[Golden]:
        # Extract contexts and source files from goldens
        contexts = []
        source_files = []
        for golden in goldens:
            if golden.context is None:
                continue
            contexts.append(golden.context)
            source_files.append(golden.source_file)

        # Extract styles from goldens if not already set
        if self.set_styling_config == False:
            example_inputs = random.sample(
                [golden.input for golden in goldens], min(len(goldens), 10)
            )
            styling_prompt = (
                ExtractionTemplate.extract_prompt_structure_from_inputs(
                    example_inputs
                )
            )
            styles = await self._a_generate_schema(
                styling_prompt, PromptStyling, self.model
            )
            styles_json = json.loads(styles.model_dump_json())
            styling_config = StylingConfig(
                **styles_json, expected_output_format=None
            )
            self.styling_config = styling_config

        # Generate goldens from scratch or from contexts if available
        if len(contexts) == 0:
            return await self.a_generate_goldens_from_scratch(
                num_goldens=len(goldens) * max_goldens_per_golden,
            )
        else:
            return await self.a_generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_golden,
                source_files=source_files,
            )

    #############################################################
    # Helper Methods for Input Generation
    #############################################################

    async def _a_generate_inputs(self, prompt: str) -> List[SyntheticData]:
        res: SyntheticDataList = await self._a_generate_schema(
            prompt, SyntheticDataList, self.model
        )
        synthetic_data_items = res.data
        return synthetic_data_items

    def _generate_inputs(self, prompt: str) -> List[SyntheticData]:
        res: SyntheticDataList = self._generate_schema(
            prompt, SyntheticDataList, self.model
        )
        synthetic_data_items = res.data
        return synthetic_data_items

    async def _a_rewrite_inputs(
        self,
        context: List[str],
        inputs: List[SyntheticData],
    ) -> Tuple[List[SyntheticData], List[float]]:
        # Evaluate input quality
        scores = []
        filtered_inputs = []
        for item in inputs:
            input = item.input
            for _ in range(self.filtration_config.max_quality_retries):
                # Evaluate synthetically generated inputs
                evaluation_prompt = FilterTemplate.evaluate_synthetic_inputs(
                    input
                )
                feedback_res: InputFeedback = await self._a_generate_schema(
                    evaluation_prompt,
                    InputFeedback,
                    self.filtration_config.critic_model,
                )
                feedback, score = feedback_res.feedback, feedback_res.score
                if (
                    score
                    >= self.filtration_config.synthetic_input_quality_threshold
                ):
                    break

                # Rewrite input if score below threshold
                rewrite_prompt = SynthesizerTemplate.rewrite_synthetic_inputs(
                    context, input, feedback
                )
                rewritten_res: RewrittenInput = await self._a_generate_schema(
                    rewrite_prompt,
                    RewrittenInput,
                    self.model,
                )
                input = rewritten_res.rewritten_input

            scores.append(score)
            filtered_inputs.append(SyntheticData(input=input))

        return filtered_inputs, scores

    def _rewrite_inputs(
        self,
        context: List[str],
        inputs: List[SyntheticData],
    ) -> Tuple[List[SyntheticData], List[float]]:
        # Evaluate input quality
        scores = []
        filtered_inputs = []
        for item in inputs:
            input = item.input
            for _ in range(self.filtration_config.max_quality_retries):
                # Evaluate synthetically generated inputs
                evaluation_prompt = FilterTemplate.evaluate_synthetic_inputs(
                    input
                )
                feedback_res: InputFeedback = self._generate_schema(
                    evaluation_prompt,
                    InputFeedback,
                    self.filtration_config.critic_model,
                )
                feedback, score = feedback_res.feedback, feedback_res.score
                if (
                    score
                    >= self.filtration_config.synthetic_input_quality_threshold
                ):
                    break

                # Rewrite input if score below threshold
                rewrite_prompt = SynthesizerTemplate.rewrite_synthetic_inputs(
                    context, input, feedback
                )
                rewritten_res: RewrittenInput = self._generate_schema(
                    rewrite_prompt,
                    RewrittenInput,
                    self.model,
                )
                input = rewritten_res.rewritten_input

            scores.append(score)
            filtered_inputs.append(SyntheticData(input=input))

        return filtered_inputs, scores

    #############################################################
    # Helper Methods for Input Evolution
    #############################################################

    def _evolve_input(
        self,
        input: str,
        num_evolutions: int,
        evolutions: Dict[Union[Evolution, PromptEvolution], float],
        context: Optional[List[str]] = None,
        progress: Optional[Progress] = None,
        pbar_evolve_input_id: Optional[int] = None,
        remove_pbar: bool = True,
    ) -> Tuple[str, List[Union[Evolution, PromptEvolution]]]:
        evolved_input = input
        evolutions_used = []
        for _ in range(num_evolutions):
            # Randomize Evolution
            evolution_type = random.choices(
                list(evolutions.keys()), list(evolutions.values())
            )[0]

            # Create Evolution Prompt
            if isinstance(evolution_type, Evolution):
                evolution_method = evolution_map[evolution_type.value]
                prompt = evolution_method(input=evolved_input, context=context)
            elif isinstance(evolution_type, PromptEvolution):
                evolution_method = prompt_evolution_map[evolution_type.value]
                prompt = evolution_method(input=evolved_input)

            # Perform Evolution
            evolved_input = self._generate(prompt)
            evolutions_used.append(evolution_type.value)

            # Update Progress
            update_pbar(progress, pbar_evolve_input_id, remove=remove_pbar)

        return evolved_input, evolutions_used

    async def _a_evolve_input(
        self,
        input: str,
        num_evolutions: int,
        evolutions: Dict[Union[Evolution, PromptEvolution], float],
        context: Optional[List[str]] = None,
        progress: Optional[Progress] = None,
        pbar_evolve_input_id: Optional[int] = None,
        remove_pbar: bool = True,
    ) -> Tuple[str, List[Union[Evolution, PromptEvolution]]]:
        evolved_input = input
        evolutions_used = []
        for _ in range(num_evolutions):
            # Randomize Evolution
            evolution_type = random.choices(
                list(evolutions.keys()), list(evolutions.values())
            )[0]

            # Create Evolution Prompt
            if isinstance(evolution_type, Evolution):
                evolution_method = evolution_map[evolution_type.value]
                prompt = evolution_method(input=evolved_input, context=context)
            elif isinstance(evolution_type, PromptEvolution):
                evolution_method = prompt_evolution_map[evolution_type.value]
                prompt = evolution_method(input=evolved_input)

            # Perform Evolution
            evolved_input = await self._a_generate(prompt)
            evolutions_used.append(evolution_type.value)

            # Update Progress
            update_pbar(progress, pbar_evolve_input_id, remove=remove_pbar)

        return evolved_input, evolutions_used

    ############################################################
    # Helper Methods for LLM Generation
    #############################################################

    def _generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
        model: DeepEvalBaseLLM,
    ) -> BaseModel:
        if is_native_model(model):
            res, cost = model.generate(prompt, schema)
            if self.synthesis_cost is not None:
                self.synthesis_cost += cost
            return res
        else:
            try:
                res = model.generate(prompt, schema=schema)
                return res
            except TypeError:
                res = model.generate(prompt)
                data = trimAndLoadJson(res, self)
                # `SyntheticDataList` is nested, so must be manually processed
                # if custom model doesn't support schema
                if schema == SyntheticDataList:
                    data_list = [SyntheticData(**item) for item in data["data"]]
                    return SyntheticDataList(data=data_list)
                else:
                    return schema(**data)

    async def _a_generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
        model: DeepEvalBaseLLM,
    ) -> BaseModel:
        if is_native_model(model):
            res, cost = await model.a_generate(prompt, schema)
            if self.synthesis_cost is not None:
                self.synthesis_cost += cost
            return res
        else:
            try:
                res = await model.a_generate(prompt, schema=schema)
                return res
            except TypeError:
                res = await model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                # `SyntheticDataList` is nested, so must be manually processed
                # if custom model doesn't support schema
                if schema == SyntheticDataList:
                    data_list = [SyntheticData(**item) for item in data["data"]]
                    return SyntheticDataList(data=data_list)
                else:
                    return schema(**data)

    def _generate(self, prompt: str) -> str:
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            if self.synthesis_cost is not None:
                self.synthesis_cost += cost
            return res
        else:
            try:
                res: Response = self.model.generate(prompt, schema=Response)
                return res.response
            except TypeError:
                res = self.model.generate(prompt)
                return res

    async def _a_generate(self, prompt: str) -> str:
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            if self.synthesis_cost is not None:
                self.synthesis_cost += cost
            return res
        else:
            try:
                res: Response = await self.model.a_generate(
                    prompt, schema=Response
                )
                return res.response
            except TypeError:
                res = await self.model.a_generate(prompt)
                return res

    #############################################################
    # Utilities
    #############################################################

    async def task_wrapper(self, sem, func, *args, **kwargs):
        async with sem:  # Acquire semaphore
            return await func(*args, **kwargs)

    def to_pandas(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )
        # Prepare data for the DataFrame
        data = []

        for golden in self.synthetic_goldens:
            # Extract basic fields
            input_text = golden.input
            expected_output = golden.expected_output
            context = golden.context
            actual_output = golden.actual_output
            retrieval_context = golden.retrieval_context
            metadata = golden.additional_metadata
            source_file = golden.source_file

            # Calculate num_context and context_length
            if context is not None:
                num_context = len(context)
                context_length = sum(len(c) for c in context)
            else:
                num_context = None
                context_length = None

            # Handle metadata
            if metadata is not None:
                evolutions = metadata.get("evolutions", None)
                synthetic_input_quality = metadata.get(
                    "synthetic_input_quality", None
                )
                context_quality = metadata.get("context_quality", None)
            else:
                evolutions = None
                synthetic_input_quality = None
                context_quality = None

            # Prepare a row for the DataFrame
            row = {
                "input": input_text,
                "actual_output": actual_output,
                "expected_output": expected_output,
                "context": context,
                "retrieval_context": retrieval_context,
                "n_chunks_per_context": num_context,
                "context_length": context_length,
                "evolutions": evolutions,
                "context_quality": context_quality,
                "synthetic_input_quality": synthetic_input_quality,
                "source_file": source_file,
            }

            # Append the row to the data list
            data.append(row)

        # Create the pandas DataFrame
        df = pd.DataFrame(data)

        # Optional: Fill NaN evolutions for better clarity
        df["evolutions"] = df["evolutions"].apply(
            lambda x: x if x is not None else "None"
        )

        return df

    def save_as(
        self,
        file_type: Literal["json", "csv", "jsonl"],
        directory: str,
        file_name: Optional[str] = None,
        quiet: bool = False,
    ) -> str:
        """Save synthetic goldens to a file.

        Args:
            file_type: Type of file to save as ('json' or 'csv').
            directory: Directory path where the file will be saved.
            file_name: Optional custom filename without extension. If provided,
                       the file will be saved as "{file_name}.{file_type}".
                       Must not contain file extension or periods.
            quiet: Optional boolean to suppress output messages about the save location.

        Returns:
            Full path to the saved file.

        Raises:
            ValueError: If file_type is invalid, no synthetic goldens exist,
            or file_name contains periods.
        """
        if str(file_type).lower() not in valid_file_types:
            raise ValueError(
                "Invalid file type. Available file types to save as: , ".join(
                    type for type in valid_file_types
                )
            )

        if file_name and "." in file_name:
            raise ValueError(
                "file_name should not contain periods or file extensions. "
                "The file extension will be added based on the file_type "
                "parameter."
            )

        if len(self.synthetic_goldens) == 0:
            raise ValueError(
                "No synthetic goldens found. Please generate goldens before saving goldens."
            )

        base_name = file_name or datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        new_filename = f"{base_name}.{file_type}"

        os.makedirs(directory, exist_ok=True)

        full_file_path = os.path.join(directory, new_filename)
        if file_type == "json":
            with open(full_file_path, "w", encoding="utf-8") as file:
                json_data = [
                    {
                        "input": golden.input,
                        "actual_output": golden.actual_output,
                        "expected_output": golden.expected_output,
                        "context": golden.context,
                        "source_file": golden.source_file,
                    }
                    for golden in self.synthetic_goldens
                ]
                json.dump(json_data, file, indent=4, ensure_ascii=False)
        elif file_type == "csv":
            with open(
                full_file_path, "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "input",
                        "actual_output",
                        "expected_output",
                        "context",
                        "source_file",
                    ]
                )
                for golden in self.synthetic_goldens:
                    writer.writerow(
                        [
                            golden.input,
                            golden.actual_output,
                            golden.expected_output,
                            "|".join(golden.context),
                            golden.source_file,
                        ]
                    )
        elif file_type == "jsonl":
            with open(full_file_path, "w", encoding="utf-8") as file:
                for golden in self.synthetic_goldens:
                    record = {
                        "input": golden.input,
                        "actual_output": golden.actual_output,
                        "expected_output": golden.expected_output,
                        "context": golden.context,
                        "source_file": golden.source_file,
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not quiet:
            print(f"Synthetic goldens saved at {full_file_path}!")

        return full_file_path
