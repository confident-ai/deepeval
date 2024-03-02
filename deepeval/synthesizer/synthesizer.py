from typing import List, Optional, Union
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
        self.synthesizer_model = self.model.get_model_name()
        self.multithreading = multithreading
        self.batch_size = batch_size
        self.synthetic_data = None

    def _synthesize_goldens(self, context: List[str], goldens: List[Golden], max_goldens_per_context: int, lock: Lock):
        prompt = SynthesizerTemplate.generate_synthetic_data(context=context, max_goldens_per_context=max_goldens_per_context)
        res = self.model(prompt)
        data = trimAndLoadJson(res)
        self.synthetic_data = [SyntheticData(**item) for item in data["data"]]
        temp_goldens : List[Golden] = []
        for data in self.synthetic_data:
            golden = Golden(
                input=data.input,
                expectedOutput=data.expected_output,
                context=context,
            )
            temp_goldens.append(golden)

        with lock:
            goldens.extend(temp_goldens)


    # TODO
    def synthesize(self, contexts: List[List[str]], max_goldens_per_context: int = 2) -> List[Golden]:
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
                        self._synthesize_goldens,
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
                prompt = SynthesizerTemplate.generate_synthetic_data(context=context, max_goldens_per_context=max_goldens_per_context)
                res = self.model(prompt)
                data = trimAndLoadJson(res)
                self.synthetic_data = [SyntheticData(**item) for item in data["data"]]

                # TODO: optional evolution

                # TODO: review synthetic data
                for data in self.synthetic_data:
                    golden = Golden(
                        input=data.input,
                        expectedOutput=data.expected_output,
                        context=context,
                    )
                    goldens.append(golden)

        return goldens

    # TODO
    def synthesize_from_docs(self, path: str):
        # Load in docs using llamaindex or langchain
        if self.multithreading:
            # Process asyncly in self.batch_size, call self.synthesize
            pass
        else:
            # Process syncly in self.batch_size, call self.synthesize
            pass
        pass

    def save(self, file_type: str, path: str):
        

        pass