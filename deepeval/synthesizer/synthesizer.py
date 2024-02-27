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
    def synthesize(
        self, context: List[str]
    ) -> List[LLMTestCase]:
        # text can ONLY be expected output, retrieval context, or context
        # (?) Optional evolution happens here
        # (?) Add option to generate n test cases based on a piece of text
        prompt = SynthesizerTemplate.generate_input_output_pairs(
            context=context
        )
        res = self.model(prompt)
        data = trimAndLoadJson(res)
        synthetic_data = [
            SyntheticData(**item) for item in data["pairs"]
        ]

        test_cases : List[LLMTestCase] = []

        for data in synthetic_data:
            test_case = LLMTestCase(input=data.input, expected_output=data.expected_output, context=context)
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
