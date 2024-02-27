from typing import List, Optional, Union
import json
from threading import Thread, Lock
from pydantic import BaseModel, Field

from deepeval.synthesizer.template import EvolutionTemplate, SynthesizerTemplate
from deepeval.models import GPTModel, DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
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

    # TODO
    def synthesize(self, context: List[str]) -> List[Golden]:
        # 1. get embeddings for context eg., [1,2,3,4,5]
        # 2. group randomly based on embedding similarity eg., [[1,2], [5,2], [4], [3,1,5]]
        # 3. supply as context, generate for each List[str]
        # 4. generation can happen in batches, and each batch can be processed in threads
        # 5. optional evolution
        # 6. optional review
        # 7. return goldens

        # TODO: logic to group and vary contexts

        # TODO: batch generation
        prompt = SynthesizerTemplate.generate_synthetic_data(context=context)
        res = self.model(prompt)
        data = trimAndLoadJson(res)
        self.synthetic_data = [SyntheticData(**item) for item in data["data"]]

        # TODO: optional evolution

        # TODO: review synthetic data

        goldens: List[Golden] = []

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
