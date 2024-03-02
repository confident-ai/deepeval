from typing import List, Optional, Union
import csv
import json
from threading import Thread, Lock
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepeval.synthesizer.template import EvolutionTemplate, SynthesizerTemplate
from deepeval.models import (
    GPTModel,
    DeepEvalBaseLLM,
    DeepEvalBaseEmbeddingModel,
)
from deepeval.utils import trimAndLoadJson
from deepeval.dataset import Golden


valid_file_types = ["csv", "json"]


class SyntheticData(BaseModel):
    input: str
    expected_output: str


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
        multithreading: bool = True,
        batch_size: int = 50,
    ):
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)

        self.embedder = embedder
        self.generator_model = self.model.get_model_name()
        self.multithreading = multithreading
        self.batch_size = batch_size
        self.synthetic_goldens : List[Golden] = []

    def _generate(
        self,
        context: List[str],
        goldens: List[Golden],
        max_goldens_per_context: int,
        lock: Lock,
    ):
        prompt = SynthesizerTemplate.generate_synthetic_data(
            context=context, max_goldens_per_context=max_goldens_per_context
        )
        res = self.model(prompt)
        data = trimAndLoadJson(res)
        synthetic_data = [SyntheticData(**item) for item in data["data"]]
        temp_goldens: List[Golden] = []
        for data in synthetic_data:
            golden = Golden(
                input=data.input,
                expectedOutput=data.expected_output,
                context=context,
            )
            temp_goldens.append(golden)

        with lock:
            goldens.extend(temp_goldens)

    # TODO
    def generate_goldens(
        self, contexts: List[List[str]], max_goldens_per_context: int = 2
    ) -> List[Golden]:
        goldens: List[Golden] = []

        # 1. get embeddings for context eg., [1,2,3,4,5]
        # 2. group randomly based on embedding similarity eg., [[1,2], [5,2], [4], [3,1,5]]
        # 3. supply as context, generate for each List[str]
        # 4. generation can happen in batches, and each batch can be processed in threads
        # 5. optional evolution
        # 6. optional review
        # 7. return goldens

        # TODO: logic to group and vary contexts

        # TODO: batch generation
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
                res = self.model(prompt)
                data = trimAndLoadJson(res)
                synthetic_data = [
                    SyntheticData(**item) for item in data["data"]
                ]

                # TODO: optional evolution

                # TODO: review synthetic data
                for data in synthetic_data:
                    golden = Golden(
                        input=data.input,
                        expectedOutput=data.expected_output,
                        context=context,
                    )
                    goldens.append(golden)

        self.synthetic_goldens.extend(goldens)

        return goldens

    # TODO
    def generate_goldens_from_docs(self, path: str):
        # Load in docs using llamaindex or langchain
        if self.multithreading:
            # Process asyncly in self.batch_size, call self.synthesize
            pass
        else:
            # Process syncly in self.batch_size, call self.synthesize
            pass
        pass


    def save(self, file_type: str, path: str):
        if file_type not in valid_file_types:
            raise ValueError(
                f"Invalid file type. Available file types to save as: {', '.join(type for type in valid_file_types)}"
            )

        if len(self.synthetic_goldens) == 0:
            raise ValueError(
                f"No synthetic goldens found. Please generate goldens before attempting to save data as {file_type}"
            )

        if file_type == 'json':
            with open(path, 'w') as file:
                json_data = [
                    {
                        'input': golden.input,
                        'expected_output': golden.expected_output,
                        'context': golden.context
                    } for golden in self.synthetic_goldens
                ]
                json.dump(json_data, file, indent=4)
        
        elif file_type == 'csv':
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['input', 'expected_output', 'context'])
                for golden in self.synthetic_goldens:
                    context_str = '|'.join(golden.context)  # Using '|' as a delimiter for context items
                    writer.writerow([golden.input, golden.expected_output, context_str])
