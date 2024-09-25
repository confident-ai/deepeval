from typing import List, Optional, Union, Tuple, Dict
from rich.console import Console
from pydantic import BaseModel
from itertools import chain
import pandas as pd
import webbrowser
import datetime
import asyncio
import random
import json
import math
import tqdm
import csv
import os

from deepeval.utils import get_or_create_event_loop, is_confident, is_in_ci_env
from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.synthesizer.synthesizer_red_team import RTSynthesizer
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.progress_context import synthesizer_progress_context
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.models import OpenAIEmbeddingModel
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset.golden import Golden
from deepeval.synthesizer.types import *
from deepeval.synthesizer.templates.template import (
    EvolutionTemplate,
    SynthesizerTemplate,
    FilterTemplate,
)
from deepeval.synthesizer.templates.template_prompt import (
    PromptEvolutionTemplate,
    PromptSynthesizerTemplate,
)
from deepeval.synthesizer.schema import (
    SyntheticData,
    SyntheticDataList,
    SQLData,
    Response,
    InputFeedback,
    RewrittenInput,
)
from deepeval.dataset.api import (
    APIDataset,
    CreateDatasetHttpResponse,
)

valid_file_types = ["csv", "json"]

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


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        critic_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
        async_mode: bool = True,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.critic_model, self.using_native_critic_model = initialize_model(
            critic_model
        )
        self.async_mode = async_mode
        self.synthetic_goldens: List[Golden] = []
        self.context_generator = None
        self.embedder = initialize_embedding_model(embedder)

    #############################################################
    # Generate Goldens from Docs
    #############################################################

    def generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = True,
        max_contexts_per_document: int = 3,
        max_goldens_per_context: int = 2,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        num_evolutions: int = 1,
        evolutions: Dict[Evolution, float] = {
            Evolution.REASONING: 1 / 7,
            Evolution.MULTICONTEXT: 1 / 7,
            Evolution.CONCRETIZING: 1 / 7,
            Evolution.CONSTRAINED: 1 / 7,
            Evolution.COMPARATIVE: 1 / 7,
            Evolution.HYPOTHETICAL: 1 / 7,
            Evolution.IN_BREADTH: 1 / 7,
        },
        use_case: UseCase = UseCase.QA,
        _send_data=True,
    ):
        # Set Embedder if not defined
        if self.embedder is None:
            self.embedder = OpenAIEmbeddingModel()

        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens = loop.run_until_complete(
                self.a_generate_goldens_from_docs(
                    document_paths,
                    include_expected_output,
                    max_contexts_per_document,
                    max_goldens_per_context,
                    chunk_size,
                    chunk_overlap,
                    num_evolutions,
                    evolutions,
                    use_case,
                )
            )
        else:
            # Generate contexts from provided docs
            if self.context_generator is None:
                self.context_generator = ContextGenerator(
                    document_paths,
                    embedder=self.embedder,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model=self.critic_model,
                )
            self.context_generator._load_docs()
            contexts, source_files, context_scores = (
                self.context_generator.generate_contexts(
                    num_context_per_document=max_contexts_per_document
                )
            )
            print(
                f"Utilizing {len(set(chain.from_iterable(contexts)))} out of {self.context_generator.total_chunks} chunks."
            )

            # Generate goldens from generated contexts
            with synthesizer_progress_context(
                "docs",
                self.model.get_model_name(),
                self.embedder.get_model_name(),
                len(contexts) * max_goldens_per_context,
                use_case,
            ) as progress_bar:
                goldens = self.generate_goldens(
                    contexts,
                    include_expected_output,
                    max_goldens_per_context,
                    num_evolutions,
                    source_files,
                    evolutions=evolutions,
                    use_case=use_case,
                    _context_scores=context_scores,
                    _progress_bar=progress_bar,
                    _send_data=False,
                )

        # Wrap-up Synthesis
        if _send_data == True:
            self._wrap_up_synthesis()
        return goldens

    async def a_generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = True,
        max_contexts_per_document: int = 3,
        max_goldens_per_context: int = 2,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        num_evolutions: int = 1,
        evolutions: Dict[Evolution, float] = {
            Evolution.REASONING: 1 / 7,
            Evolution.MULTICONTEXT: 1 / 7,
            Evolution.CONCRETIZING: 1 / 7,
            Evolution.CONSTRAINED: 1 / 7,
            Evolution.COMPARATIVE: 1 / 7,
            Evolution.HYPOTHETICAL: 1 / 7,
            Evolution.IN_BREADTH: 1 / 7,
        },
        use_case: UseCase = UseCase.QA,
    ):
        # Set Embedder if not defined
        if self.embedder is None:
            self.embedder = OpenAIEmbeddingModel()

        # Generate contexts from provided docs
        if self.context_generator is None:
            self.context_generator = ContextGenerator(
                document_paths,
                embedder=self.embedder,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model=self.critic_model,
            )
        await self.context_generator._a_load_docs()

        contexts, source_files, context_scores = (
            await self.context_generator.a_generate_contexts(
                num_context_per_document=max_contexts_per_document
            )
        )
        print(
            f"Utilizing {len(set(chain.from_iterable(contexts)))} out of {self.context_generator.total_chunks} chunks."
        )

        # Generate goldens from generated contexts
        with synthesizer_progress_context(
            method="docs",
            evaluation_model=self.model.get_model_name(),
            embedder=self.embedder.get_model_name(),
            max_generations=len(contexts) * max_goldens_per_context,
            use_case=use_case.value,
        ) as progress_bar:
            goldens = await self.a_generate_goldens(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                num_evolutions=num_evolutions,
                source_files=source_files,
                evolutions=evolutions,
                use_case=use_case,
                _context_scores=context_scores,
                _progress_bar=progress_bar,
            )
        self.synthetic_goldens.extend(goldens)
        return goldens

    #############################################################
    # Generate Goldens from Contexts
    #############################################################

    def generate_goldens(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        num_evolutions: int = 1,
        source_files: Optional[List[str]] = None,
        evolutions: Dict[Evolution, float] = {
            Evolution.REASONING: 1 / 7,
            Evolution.MULTICONTEXT: 1 / 7,
            Evolution.CONCRETIZING: 1 / 7,
            Evolution.CONSTRAINED: 1 / 7,
            Evolution.COMPARATIVE: 1 / 7,
            Evolution.HYPOTHETICAL: 1 / 7,
            Evolution.IN_BREADTH: 1 / 7,
        },
        use_case: UseCase = UseCase.QA,
        _context_scores: Optional[List[float]] = None,
        _progress_bar: Optional[tqdm.std.tqdm] = None,
        _send_data: bool = True,
    ) -> List[Golden]:
        # Intialize Goldens as an empty list
        goldens: List[Golden] = []

        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens.extend(
                loop.run_until_complete(
                    self.a_generate_goldens(
                        contexts=contexts,
                        include_expected_output=include_expected_output,
                        max_goldens_per_context=max_goldens_per_context,
                        num_evolutions=num_evolutions,
                        source_files=source_files,
                        evolutions=evolutions,
                        use_case=use_case,
                    )
                )
            )
        else:
            if use_case == UseCase.QA:
                with synthesizer_progress_context(
                    method="default",
                    evaluation_model=self.model.get_model_name(),
                    embedder=None,
                    max_generations=len(contexts) * max_goldens_per_context,
                    use_case=use_case.value,
                    progress_bar=_progress_bar,
                    async_mode=False,
                ) as progress_bar:
                    for i, context in enumerate(contexts):

                        # Generate inputs
                        prompt = SynthesizerTemplate.generate_synthetic_inputs(
                            context=context,
                            max_goldens_per_context=max_goldens_per_context,
                        )
                        synthetic_inputs = self._generate_inputs(prompt)

                        # Qualify inputs
                        qualified_synthetic_inputs: List[SyntheticData]
                        scores: List[float]
                        qualified_synthetic_inputs, scores = (
                            self._rewrite_inputs(context, synthetic_inputs)
                        )
                        for j, data in enumerate(qualified_synthetic_inputs):

                            # Evolve input
                            evolved_input, evolutions_used = self._evolve_input(
                                input=data.input,
                                context=context,
                                num_evolutions=num_evolutions,
                                evolutions=evolutions,
                                progress_bar=progress_bar,
                            )

                            # Synthesize Golden
                            golden = Golden(
                                input=evolved_input,
                                context=context,
                                source_file=(
                                    source_files[i]
                                    if source_files is not None
                                    else None
                                ),
                                additional_metadata={
                                    "evolutions": evolutions_used,
                                    "synthetic_input_quality": scores[j],
                                    "context_quality": (
                                        _context_scores[i]
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
                                )
                                res = self._generate(prompt)
                                golden.expected_output = res
                            goldens.append(golden)

            elif use_case == UseCase.TEXT2SQL:
                with synthesizer_progress_context(
                    method="default",
                    evaluation_model=self.model.get_model_name(),
                    embedder=None,
                    max_generations=len(contexts) * max_goldens_per_context,
                    use_case=use_case.value,
                    progress_bar=_progress_bar,
                    async_mode=False,
                ) as progress_bar:
                    for i, context in enumerate(contexts):

                        # Generate inputs
                        prompt = SynthesizerTemplate.generate_text2sql_inputs(
                            context=context,
                            max_goldens_per_context=max_goldens_per_context,
                        )
                        synthetic_data = self._generate_inputs(prompt)
                        for data in synthetic_data:

                            # Synthesize Golden
                            golden = Golden(
                                input=data.input,
                                context=context,
                            )

                            # Generate expected output
                            if include_expected_output:
                                prompt = SynthesizerTemplate.generate_text2sql_expected_output(
                                    input=golden.input,
                                    context="\n".join(golden.context),
                                )
                                expected_output: SQLData = (
                                    self._generate_schema(
                                        prompt,
                                        SQLData,
                                        self.model,
                                        self.using_native_model,
                                    )
                                )
                                golden.expected_output = expected_output.sql

                            goldens.append(golden)
                            if progress_bar is not None:
                                progress_bar.update(1)

        # Wrap-up Synthesis
        self.synthetic_goldens.extend(goldens)
        if _send_data == True:
            self._wrap_up_synthesis()
        return goldens

    async def a_generate_goldens(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        num_evolutions: int = 1,
        source_files: Optional[List[str]] = None,
        evolutions: Dict[Evolution, float] = {
            Evolution.REASONING: 1 / 7,
            Evolution.MULTICONTEXT: 1 / 7,
            Evolution.CONCRETIZING: 1 / 7,
            Evolution.CONSTRAINED: 1 / 7,
            Evolution.COMPARATIVE: 1 / 7,
            Evolution.HYPOTHETICAL: 1 / 7,
            Evolution.IN_BREADTH: 1 / 7,
        },
        use_case: UseCase = UseCase.QA,
        _context_scores: Optional[List[float]] = None,
        _progress_bar: Optional[tqdm.std.tqdm] = None,
    ) -> List[Golden]:
        goldens: List[Golden] = []
        if use_case == UseCase.QA:
            with synthesizer_progress_context(
                method="default",
                evaluation_model=self.model.get_model_name(),
                embedder=None,
                max_generations=len(contexts) * max_goldens_per_context,
                use_case=use_case.value,
                progress_bar=_progress_bar,
                async_mode=True,
            ) as progress_bar:
                tasks = [
                    self._a_generate_from_context(
                        context=context,
                        goldens=goldens,
                        include_expected_output=include_expected_output,
                        max_goldens_per_context=max_goldens_per_context,
                        num_evolutions=num_evolutions,
                        source_files=source_files,
                        index=index,
                        evolutions=evolutions,
                        progress_bar=progress_bar,
                        context_scores=_context_scores,
                    )
                    for index, context in enumerate(contexts)
                ]
                await asyncio.gather(*tasks)

        elif use_case == UseCase.TEXT2SQL:
            with synthesizer_progress_context(
                method="default",
                evaluation_model=self.model.get_model_name(),
                embedder=None,
                max_generations=len(contexts) * max_goldens_per_context,
                use_case=use_case.value,
                progress_bar=_progress_bar,
                async_mode=True,
            ) as progress_bar:
                tasks = [
                    self._a_generate_text_to_sql_from_context(
                        context=context,
                        goldens=goldens,
                        include_expected_output=include_expected_output,
                        max_goldens_per_context=max_goldens_per_context,
                        progress_bar=progress_bar,
                    )
                    for context in contexts
                ]
                await asyncio.gather(*tasks)

        return goldens

    async def _a_generate_from_context(
        self,
        context: List[str],
        goldens: List[Golden],
        include_expected_output: bool,
        max_goldens_per_context: int,
        num_evolutions: int,
        source_files: Optional[List[str]],
        index: int,
        evolutions: List[Evolution],
        progress_bar: tqdm.std.tqdm,
        context_scores: Optional[List[float]] = None,
    ):
        # Generate inputs
        prompt = SynthesizerTemplate.generate_synthetic_inputs(
            context=context, max_goldens_per_context=max_goldens_per_context
        )
        synthetic_inputs: List[SyntheticData] = await self._a_generate_inputs(
            prompt
        )

        # Qualify inputs
        qualified_synthetic_inputs: List[SyntheticData]
        scores: List[float]
        qualified_synthetic_inputs, scores = await self._a_rewrite_inputs(
            context, synthetic_inputs
        )
        for i, data in enumerate(qualified_synthetic_inputs):

            # Evolve input
            evolved_input, evolutions_used = await self._a_evolve_input(
                input=data.input,
                context=context,
                num_evolutions=num_evolutions,
                evolutions=evolutions,
            )

            # Generate expected output
            expected_output = None
            if include_expected_output:
                expected_output_prompt = (
                    SynthesizerTemplate.generate_synthetic_expected_output(
                        input=evolved_input, context="\n".join(context)
                    )
                )
                expected_output = await self._a_generate(expected_output_prompt)

            # Synthesize Golden
            golden = Golden(
                input=evolved_input,
                context=context,
                expected_output=expected_output,
                source_file=(
                    source_files[index] if source_files is not None else None
                ),
                additional_metadata={
                    "evolutions": evolutions_used,
                    "synthetic_input_quality": scores[i],
                    "context_quality": (
                        context_scores[i]
                        if context_scores is not None
                        else None
                    ),
                },
            )
            goldens.append(golden)

            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)

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
                    prompt, SQLData, self.model, self.using_native_model
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
        subject: str,
        task: str,
        output_format: str,
        num_initial_goldens: int,
        num_evolutions: int = 1,
        evolutions: Dict[PromptEvolution, float] = {
            PromptEvolution.REASONING: 1 / 6,
            PromptEvolution.CONCRETIZING: 1 / 6,
            PromptEvolution.CONSTRAINED: 1 / 6,
            PromptEvolution.COMPARATIVE: 1 / 6,
            PromptEvolution.HYPOTHETICAL: 1 / 6,
            PromptEvolution.IN_BREADTH: 1 / 6,
        },
    ) -> List[Golden]:
        goldens: List[Golden] = []
        with synthesizer_progress_context(
            method="Scratch",
            evaluation_model=self.model.get_model_name(),
            embedder=None,
            max_generations=num_initial_goldens,
            use_case=None,
            progress_bar=None,
            async_mode=True,
        ) as progress_bar:

            # Generate inputs
            prompt: List = PromptSynthesizerTemplate.generate_synthetic_prompts(
                subject=subject,
                task=task,
                output_format=output_format,
                num_initial_goldens=num_initial_goldens,
            )
            synthetic_data = self._generate_inputs(prompt)

            # Evolve inputs
            tasks = [
                self._a_evolve_input(
                    input=data.input,
                    num_evolutions=num_evolutions,
                    evolutions=evolutions,
                    progress_bar=progress_bar,
                )
                for data in synthetic_data
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
            return goldens

    def generate_goldens_from_scratch(
        self,
        subject: str,
        task: str,
        output_format: str,
        num_initial_goldens: int,
        num_evolutions: int = 1,
        evolutions: Dict[PromptEvolution, float] = {
            PromptEvolution.REASONING: 1 / 6,
            PromptEvolution.CONCRETIZING: 1 / 6,
            PromptEvolution.CONSTRAINED: 1 / 6,
            PromptEvolution.COMPARATIVE: 1 / 6,
            PromptEvolution.HYPOTHETICAL: 1 / 6,
            PromptEvolution.IN_BREADTH: 1 / 6,
        },
        _send_data: bool = True,
    ) -> List[Golden]:
        goldens: List[Golden] = []
        if self.async_mode:
            loop = get_or_create_event_loop()
            goldens.extend(
                loop.run_until_complete(
                    self.a_generate_goldens_from_scratch(
                        subject=subject,
                        task=task,
                        output_format=output_format,
                        num_evolutions=num_evolutions,
                        num_initial_goldens=num_initial_goldens,
                        evolutions=evolutions,
                    )
                )
            )
        else:
            with synthesizer_progress_context(
                method="Scratch",
                evaluation_model=self.model.get_model_name(),
                embedder=None,
                max_generations=num_initial_goldens,
                use_case=None,
                progress_bar=None,
                async_mode=False,
            ) as progress_bar:

                # Generate inputs
                prompt: List = (
                    PromptSynthesizerTemplate.generate_synthetic_prompts(
                        subject=subject,
                        task=task,
                        output_format=output_format,
                        num_initial_goldens=num_initial_goldens,
                    )
                )
                synthetic_data = self._generate_inputs(prompt)

                # Evolve inputs
                for data in synthetic_data:
                    evolved_prompt, evolutions_used = self._evolve_input(
                        input=data.input,
                        num_evolutions=num_evolutions,
                        evolutions=evolutions,
                        progress_bar=progress_bar,
                    )

                    # Synthesize Goldens
                    golden = Golden(
                        input=evolved_prompt,
                        additional_metadata={"evolutions": evolutions_used},
                    )
                    goldens.append(golden)

        # Wrap up Synthesis
        self.synthetic_goldens.extend(goldens)
        if _send_data == True:
            self._wrap_up_synthesis()
        return goldens

    #############################################################
    # Helper Methods for Input Generation
    #############################################################

    async def _a_generate_inputs(self, prompt: str) -> List[SyntheticData]:
        res: SyntheticDataList = await self._a_generate_schema(
            prompt, SyntheticDataList, self.model, self.using_native_model
        )
        synthetic_data_items = res.data
        return synthetic_data_items

    def _generate_inputs(self, prompt: str) -> List[SyntheticData]:
        res: SyntheticDataList = self._generate_schema(
            prompt, SyntheticDataList, self.model, self.using_native_model
        )
        synthetic_data_items = res.data
        return synthetic_data_items

    async def _a_rewrite_inputs(
        self,
        context: List[str],
        inputs: List[SyntheticData],
        max_retries: int = 3,
        threshold: int = 0.5,
    ) -> Tuple[List[SyntheticData], List[float]]:

        # Evaluate input quality
        scores = []
        filtered_inputs = []
        for item in inputs:
            input = item.input
            for _ in range(max_retries):

                # Evaluate synthetically generated inputs
                evaluation_prompt = FilterTemplate.evaluate_synthetic_inputs(
                    input
                )
                feedback_res: InputFeedback = await self._a_generate_schema(
                    evaluation_prompt,
                    InputFeedback,
                    self.critic_model,
                    self.using_native_critic_model,
                )
                feedback, score = feedback_res.feedback, feedback_res.score
                if score >= threshold:
                    break

                # Rewrite input if score below threshold
                rewrite_prompt = SynthesizerTemplate.rewrite_synthetic_inputs(
                    context, input, feedback
                )
                rewritten_res: RewrittenInput = await self._a_generate_schema(
                    rewrite_prompt,
                    RewrittenInput,
                    self.model,
                    self.using_native_model,
                )
                input = rewritten_res.rewritten_input

            scores.append(score)
            filtered_inputs.append(SyntheticData(input=input))

        return filtered_inputs, scores

    def _rewrite_inputs(
        self,
        context: List[str],
        inputs: List[SyntheticData],
        max_retries: int = 3,
        threshold: int = 0.5,
    ) -> Tuple[List[SyntheticData], List[float]]:

        # Evaluate input quality
        scores = []
        filtered_inputs = []
        for item in inputs:
            input = item.input
            for _ in range(max_retries):

                # Evaluate synthetically generated inputs
                evaluation_prompt = FilterTemplate.evaluate_synthetic_inputs(
                    input
                )
                feedback_res: InputFeedback = self._generate_schema(
                    evaluation_prompt,
                    InputFeedback,
                    self.critic_model,
                    self.using_native_critic_model,
                )
                feedback, score = feedback_res.feedback, feedback_res.score
                if score >= threshold:
                    break

                # Rewrite input if score below threshold
                rewrite_prompt = SynthesizerTemplate.rewrite_synthetic_inputs(
                    context, input, feedback
                )
                rewritten_res: RewrittenInput = self._generate_schema(
                    rewrite_prompt,
                    RewrittenInput,
                    self.model,
                    self.using_native_model,
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
        progress_bar: Optional[tqdm.std.tqdm] = None,
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
        if progress_bar:
            progress_bar.update(1)

        return evolved_input, evolutions_used

    async def _a_evolve_input(
        self,
        input: str,
        num_evolutions: int,
        evolutions: Dict[Union[Evolution, PromptEvolution], float],
        context: Optional[List[str]] = None,
        progress_bar: Optional[tqdm.std.tqdm] = None,
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
        if progress_bar:
            progress_bar.update(1)

        return evolved_input, evolutions_used

    ############################################################
    # Helper Methods for LLM Generation
    #############################################################

    def _generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
        model: DeepEvalBaseLLM,
        using_native_model: bool,
    ) -> Tuple[str, float]:
        if using_native_model:
            res, _ = model.generate(prompt)
            data = trimAndLoadJson(res, self)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)
        else:
            try:
                res: schema = model.generate(prompt, schema=schema)
                return res
            except TypeError:
                res = model.generate(prompt)
                data = trimAndLoadJson(res, self)
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
        using_native_model: bool,
    ) -> Tuple[str, float]:
        if using_native_model:
            res, _ = await model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)
        else:
            try:
                res: schema = await model.a_generate(prompt, schema=schema)
                return res
            except TypeError:
                res = await model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                if schema == SyntheticDataList:
                    data_list = [SyntheticData(**item) for item in data["data"]]
                    return SyntheticDataList(data=data_list)
                else:
                    return schema(**data)

    def _generate(self, prompt: str) -> Tuple[str, str]:
        if self.using_native_model:
            res, _ = self.model.generate(prompt)
            return res
        else:
            try:
                res: Response = self.model.generate(prompt, schema=Response)
                return res.response
            except TypeError:
                res = self.model.generate(prompt)
                return res

    async def _a_generate(self, prompt: str) -> Tuple[str, str]:
        if self.using_native_model:
            res, _ = await self.model.a_generate(prompt)
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

    def to_pandas(self):
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

    def _wrap_up_synthesis(self):
        console = Console()
        if is_confident():
            alias = input("Enter the dataset alias: ").strip()
            if len(self.synthetic_goldens) == 0:
                raise ValueError(
                    "Unable to push empty dataset to Confident AI. There must be at least one dataset or golden data entry."
                )
            try:
                console.print(
                    "Sending a large dataset to Confident AI. This might take a bit longer than usual..."
                )
                goldens = self.synthetic_goldens
                api_dataset = APIDataset(alias=alias, goldens=goldens)
                try:
                    body = api_dataset.model_dump(
                        by_alias=True, exclude_none=True
                    )
                except AttributeError:
                    body = api_dataset.dict(by_alias=True, exclude_none=True)
                api = Api()
                result = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.DATASET_ENDPOINT,
                    body=body,
                )
                if result:
                    response = CreateDatasetHttpResponse(link=result["link"])
                    link = response.link
                    console.print(
                        f"✅ Dataset successfully pushed to Confident AI! View at [link={link}]{link}[/link]"
                    )
                    webbrowser.open(link)
            except Exception as e:
                message = f"Unexpected error when sending the dataset. Incomplete dataset push is available at {link if 'link' in locals() else 'N/A'}."
                raise Exception(message) from e
        else:
            raise Exception(
                "To push a dataset to Confident AI, please run `deepeval login` first."
            )

    def save_as(self, file_type: str, directory: str) -> str:
        if file_type not in valid_file_types:
            raise ValueError(
                f"Invalid file type. Available file types to save as: {', '.join(type for type in valid_file_types)}"
            )
        if len(self.synthetic_goldens) == 0:
            raise ValueError(
                f"No synthetic goldens found. Please generate goldens before attempting to save data as {file_type}"
            )
        new_filename = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f".{file_type}"
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_file_path = os.path.join(directory, new_filename)
        if file_type == "json":
            with open(full_file_path, "w") as file:
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
                json.dump(json_data, file, indent=4)
        elif file_type == "csv":
            with open(full_file_path, "w", newline="") as file:
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
        print(f"Synthetic goldens saved at {full_file_path}!")
        return full_file_path
