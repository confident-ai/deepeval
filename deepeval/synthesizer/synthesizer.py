from typing import List, Optional, Union
import os
import csv
import json
from threading import Lock
from pydantic import BaseModel
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import math

from deepeval.synthesizer.template import EvolutionTemplate, SynthesizerTemplate
from deepeval.synthesizer.context_generator import ContextGenerator
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.models import DeepEvalBaseLLM
from deepeval.progress_context import synthesizer_progress_context
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.dataset.golden import Golden
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.models import OpenAIEmbeddingModel

valid_file_types = ["csv", "json"]


class SyntheticData(BaseModel):
    input: str


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
        multithreading: bool = True,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.multithreading = multithreading
        self.synthetic_goldens: List[Golden] = []
        self.context_generator = None
        self.embedder = initialize_embedding_model(embedder)

    def evolve(
        self,
        queries: List[str],
        num_evolutions: int,
        enable_breadth_evolve: bool = True,
    ) -> List[str]:
        # List of method references from EvolutionTemplate
        evolution_methods = [
            EvolutionTemplate.reasoning_evolution,
            EvolutionTemplate.multi_context_evolution,
            EvolutionTemplate.concretizing_evolution,
            EvolutionTemplate.constrained_evolution,
            EvolutionTemplate.comparative_question_evolution,
            EvolutionTemplate.hypothetical_scenario_evolution,
        ]
        if enable_breadth_evolve:
            evolution_methods.append(EvolutionTemplate.in_breadth_evolution)

        evolved_queries = [q for q in queries]
        for _ in range(num_evolutions):
            for i in range(len(queries)):
                evolution_method = random.choice(evolution_methods)
                prompt = evolution_method(
                    input=queries[i - 1],
                    context="make the input more complex, drawing from your knowledge base",
                )
                if self.using_native_model:
                    evolved_query, cost = self.model.generate(prompt)
                else:
                    evolved_query = self.model.generate(prompt)
                evolved_queries.append(evolved_query)

        return evolved_queries

    def _evolve_text(
        self,
        text,
        context: List[str],
        num_evolutions: int,
        enable_breadth_evolve: bool,
    ) -> List[str]:
        # List of method references from EvolutionTemplate
        evolution_methods = [
            EvolutionTemplate.reasoning_evolution,
            EvolutionTemplate.multi_context_evolution,
            EvolutionTemplate.concretizing_evolution,
            EvolutionTemplate.constrained_evolution,
            EvolutionTemplate.comparative_question_evolution,
            EvolutionTemplate.hypothetical_scenario_evolution,
        ]
        if enable_breadth_evolve:
            evolution_methods.append(EvolutionTemplate.in_breadth_evolution)

        evolved_text = text
        for _ in range(num_evolutions):
            evolution_method = random.choice(evolution_methods)
            prompt = evolution_method(input=evolved_text, context=context)
            if self.using_native_model:
                evolved_text, cost = self.model.generate(prompt)
            else:
                evolved_text = self.model.generate(prompt)

        return evolved_text

    def _generate(
        self,
        context: List[str],
        goldens: List[Golden],
        include_expected_output: bool,
        max_goldens_per_context: int,
        lock: Lock,
        num_evolutions: int,
        enable_breadth_evolve: bool,
        source_files: Optional[List[str]],
        index: int,
    ):
        prompt = SynthesizerTemplate.generate_synthetic_inputs(
            context=context, max_goldens_per_context=max_goldens_per_context
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        synthetic_data = [SyntheticData(**item) for item in data["data"]]

        temp_goldens: List[Golden] = []
        for data in synthetic_data:
            evolved_input = self._evolve_text(
                data.input,
                context=context,
                num_evolutions=num_evolutions,
                enable_breadth_evolve=enable_breadth_evolve,
            )
            source_file = (
                source_files[index] if source_files is not None else None
            )
            golden = Golden(
                input=evolved_input, context=context, sourceFile=source_file
            )

            if include_expected_output:
                prompt = SynthesizerTemplate.generate_synthetic_expected_output(
                    input=golden.input, context="\n".join(golden.context)
                )
                if self.using_native_model:
                    res, cost = self.model.generate(prompt)
                else:
                    res = self.model.generate(prompt)

                golden.expected_output = res

            temp_goldens.append(golden)

        with lock:
            goldens.extend(temp_goldens)

    def generate_goldens(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = False,
        max_goldens_per_context: int = 2,
        num_evolutions: int = 1,
        enable_breadth_evolve: bool = False,
        source_files: Optional[List[str]] = None,
        _show_indicator: bool = True,
    ) -> List[Golden]:
        with synthesizer_progress_context(
            self.model.get_model_name(),
            None,
            contexts * max_goldens_per_context,
            _show_indicator,
        ):
            goldens: List[Golden] = []
            if self.multithreading:
                lock = Lock()

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            self._generate,
                            context,
                            goldens,
                            include_expected_output,
                            max_goldens_per_context,
                            lock,
                            num_evolutions,
                            enable_breadth_evolve,
                            source_files,
                            index,
                        ): context
                        for index, context in enumerate(contexts)
                    }

                    for future in as_completed(futures):
                        future.result()
            else:
                for i, context in enumerate(contexts):
                    prompt = SynthesizerTemplate.generate_synthetic_inputs(
                        context=context,
                        max_goldens_per_context=max_goldens_per_context,
                    )

                    if self.using_native_model:
                        res, cost = self.model.generate(prompt)
                    else:
                        res = self.model.generate(prompt)

                    data = trimAndLoadJson(res)
                    synthetic_data = [
                        SyntheticData(**item) for item in data["data"]
                    ]
                    for data in synthetic_data:
                        evolved_input = self._evolve_text(
                            data.input,
                            context=context,
                            num_evolutions=num_evolutions,
                            enable_breadth_evolve=enable_breadth_evolve,
                        )
                        source_file = (
                            source_files[i]
                            if source_files is not None
                            else None
                        )
                        golden = Golden(
                            input=evolved_input,
                            context=context,
                            source_file=source_file,
                        )

                        if include_expected_output:
                            prompt = SynthesizerTemplate.generate_synthetic_expected_output(
                                input=golden.input,
                                context="\n".join(golden.context),
                            )
                            if self.using_native_model:
                                res, cost = self.model.generate(prompt)
                            else:
                                res = self.model.generate(prompt)

                            golden.expected_output = res

                        goldens.append(golden)

            self.synthetic_goldens.extend(goldens)
            return goldens

    def generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = False,
        max_goldens_per_document: int = 5,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        num_evolutions: int = 1,
        enable_breadth_evolve: bool = False,
    ):
        if self.embedder is None:
            self.embedder = OpenAIEmbeddingModel()

        with synthesizer_progress_context(
            self.model.get_model_name(),
            self.embedder.get_model_name(),
            max_goldens_per_document * len(document_paths),
        ):
            if self.context_generator is None:
                self.context_generator = ContextGenerator(
                    document_paths,
                    embedder=self.embedder,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    multithreading=self.multithreading,
                )

            max_goldens_per_context = 2
            if max_goldens_per_document < max_goldens_per_context:
                max_goldens_per_context = 1

            num_context = math.floor(
                max_goldens_per_document / max_goldens_per_context
            )

            contexts, source_files = self.context_generator.generate_contexts(
                num_context=num_context
            )

            return self.generate_goldens(
                contexts,
                include_expected_output,
                max_goldens_per_context,
                num_evolutions,
                enable_breadth_evolve,
                source_files,
                _show_indicator=False,
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
