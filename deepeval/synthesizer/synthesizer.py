from typing import List, Optional, Union
import json
from threading import Thread, Lock
from pydantic import BaseModel, Field

from deepeval.synthesizer.template import EvolutionTemplate, SynthesizerTemplate
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.utils import trimAndLoadJson


class SyntheticData(BaseModel):
    input: str
    expected_output: str = Field(default=None)


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        multithreading: bool = True,
        batch_size: int = 50,
    ):
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.synthesizer_model = self.model.get_model_name()
        self.multithreading = multithreading
        self.batch_size = batch_size

    # TODO
    def synthesize(self, context: List[str]) -> List[LLMTestCase]:
        # 1. get embeddings for context eg., [1,2,3,4,5]
        # 2. group randomly based on embedding similarity eg., [[1,2], [5,2], [4], [3,1,5]]
        # 3. supply as context, generate for each List[str]
        # 4. generation can happen in batches, and each batch can be processed in threads
        # 5. optional evolution
        # 6. return test cases

        prompt = SynthesizerTemplate.generate_synthetic_data(context=context)
        res = self.model(prompt)
        data = trimAndLoadJson(res)
        synthetic_data = [SyntheticData(**item) for item in data["data"]]

        test_cases: List[LLMTestCase] = []

        for data in synthetic_data:
            test_case = LLMTestCase(
                input=data.input,
                expected_output=data.expected_output,
                context=context,
            )
            test_cases.append(test_case)

        return test_cases

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
