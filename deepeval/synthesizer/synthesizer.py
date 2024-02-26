from typing import List, Optional, Union

from deepeval.synthesizer.template import EvolutionTemplate
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class Synthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        multithreading: bool = True,
        batch_size: int = 50
    ):
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.synthesizer_model = self.model.get_model_name()
        self.multithreading = multithreading
        self.batch_size = batch_size

    # TODO
    def synthesize(self, text: Union[str, List[str]], source_param: LLMTestCaseParams) -> List[LLMTestCase]:
        # text can ONLY be expected output, retrieval context, or context
        # (?) Optional evolution happens here
        # (?) Add option to generate n test cases based on a piece of text
        pass

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
