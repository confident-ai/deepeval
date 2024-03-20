from typing import List, Optional, Union
import os
import csv
import json
from threading import Lock
from pydantic import BaseModel
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepeval.synthesizer.template import EvolutionTemplate, SynthesizerTemplate
from deepeval.synthesizer.context_generator import ContextGenerator
from deepeval.models import (
    GPTModel,
    DeepEvalBaseLLM,
    DeepEvalBaseEmbeddingModel,
)
from deepeval.progress_context import synthesizer_progress_context
from deepeval.utils import trimAndLoadJson
from deepeval.dataset.golden import Golden
import random

valid_file_types = ["csv", "json"]


class SyntheticData(BaseModel):
    input: str


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        # embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
        multithreading: bool = True,
        # batch_size: int = 50,
    ):
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)

        # self.embedder = embedder
        self.generator_model = self.model.get_model_name()
        self.multithreading = multithreading
        # self.batch_size = batch_size
        self.synthetic_goldens: List[Golden] = []

    def _evolve_text(self, 
                     text, 
                     context: List[str], 
                     num_evolutions: int, 
                     enable_breadth_evolve: bool) -> List[str]:
        # List of method references from EvolutionTemplate
        evolution_methods = [
            EvolutionTemplate.reasoning_evolution,
            EvolutionTemplate.multi_context_evolution,
            EvolutionTemplate.concretizing_evolution,
            EvolutionTemplate.constrained_evolution,
            EvolutionTemplate.comparative_question_evolution,
            EvolutionTemplate.hypothetical_scenario_evolution
        ]
        if enable_breadth_evolve:
            evolution_method.append(EvolutionTemplate.in_breadth_evolution)
        
        evolved_text = text
        for _ in range(num_evolutions):
            evolution_method = random.choice(evolution_methods)
            prompt = evolution_method(input=evolved_text, context=context)
            evolved_text = self.model.generate(prompt)

        return evolved_text
    
    def _generate(
        self,
        context: List[str],
        goldens: List[Golden],
        max_goldens_per_context: int,
        lock: Lock,
        num_evolutions: int,
        enable_breadth_evolve:bool
    ):
        prompt = SynthesizerTemplate.generate_synthetic_data(
            context=context, max_goldens_per_context=max_goldens_per_context
        )
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        synthetic_data = [SyntheticData(**item) for item in data["data"]]
        temp_goldens: List[Golden] = []
        for data in synthetic_data:
            # TODO: evolution
            # Note: skip multithreading for now
            evolved_input = self._evolve_text(data.input, 
                                              context=context, 
                                              num_evolutions=num_evolutions,
                                              enable_breadth_evolve=enable_breadth_evolve)
            golden = Golden(input=evolved_input, context=context)
            temp_goldens.append(golden)

        with lock:
            goldens.extend(temp_goldens)

    # TODO
    def generate_goldens(
        self, contexts: List[List[str]], 
        max_goldens_per_context: int = 2, 
        num_evolutions: int = 1,
        enable_breadth_evolve: bool = False,
    ) -> List[Golden]:
        with synthesizer_progress_context(self.generator_model):
            goldens: List[Golden] = []

            # 1. get embeddings for context eg., [1,2,3,4,5]
            # 2. group randomly based on embedding similarity eg., [[1,2], [5,2], [4], [3,1,5]]
            # 3. supply as context, generate for each List[str]
            # 4. generation can happen in batches, and each batch can be processed in threads
            # 5. optional evolution
            # 6. optional review
            # 7. return goldens

            # TODO: logic to group and vary contexts

            if self.multithreading:
                lock = Lock()

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            self._generate,
                            context,
                            goldens,
                            max_goldens_per_context,
                            lock,
                            num_evolutions,
                            enable_breadth_evolve
                        ): context
                        for context in contexts
                    }

                    for future in as_completed(futures):
                        future.result()
            else:
                for context in contexts:
                    prompt = SynthesizerTemplate.generate_synthetic_data(
                        context=context,
                        max_goldens_per_context=max_goldens_per_context,
                    )
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res)
                    synthetic_data = [
                        SyntheticData(**item) for item in data["data"]
                    ]
                    for data in synthetic_data:
                         evolved_input = self._evolve_text(data.input, 
                                                           context=context, 
                                                           num_evolutions=num_evolutions,
                                                           enable_breadth_evolve=enable_breadth_evolve)
                         golden = Golden(input=evolved_input, context=context)
                         goldens.append(golden)

            self.synthetic_goldens.extend(goldens)

            return goldens

    # TODO
    def generate_goldens_from_docs(
            self, 
            paths: List[str],
            num_context:int,
            chunk_size:int=1024, 
            chunk_overlap:int=0,
            max_goldens_per_context: int = 2, 
            num_evolutions: int = 1,
            enable_breadth_evolve: bool = False
            ):
        
        contexts = None
        if self.multithreading:
            cg = ContextGenerator(paths, 
                                  chunk_size, 
                                  chunk_overlap, 
                                  fast_mode=False, 
                                  multithreading=True)
            contexts = cg.generate_contexts(num_context=num_context)
        else:
            cg = ContextGenerator(paths, 
                                  chunk_size, 
                                  chunk_overlap, 
                                  fast_mode=False, 
                                  multithreading=False)
            contexts = cg.generate_contexts(num_context=num_context)

        goldens = self.generate_goldens(contexts, 
                                        max_goldens_per_context, 
                                        num_evolutions, 
                                        enable_breadth_evolve)
        return goldens

    def save_as(self, file_type: str, directory: str):
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
                    }
                    for golden in self.synthetic_goldens
                ]
                json.dump(json_data, file, indent=4)

        elif file_type == "csv":
            with open(full_file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["input", "actual_output", "expected_output", "context"]
                )
                for golden in self.synthetic_goldens:
                    context_str = "|".join(golden.context)
                    writer.writerow(
                        [
                            golden.input,
                            golden.actual_output,
                            golden.expected_output,
                            context_str,
                        ]
                    )

        print(f"Synthetic goldens saved at {full_file_path}!")


####################################################
################# Example Usage ###################
####################################################

'''
if __name__ == "__main__":
    synthesizer = Synthesizer()
    paths = ["example_data/docx_example.docx"]
    goldens = synthesizer.generate_goldens_from_docs(paths=paths, 
                                                     chunk_size=100, 
                                                     num_context=5, 
                                                     max_goldens_per_context=2, 
                                                     num_evolutions=2)
    print(goldens)
'''