from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.utils import initialize_model
from typing import List, Optional, Any, Union
from deepeval.models import DeepEvalBaseLLM

######################################################
# Schema #############################################
######################################################

class BinaryClassificationVerdict():
    verdict: bool

######################################################
# Base Node ##########################################
######################################################

class StoppingClassificationScore():
    classification: Union[str, bool]
    score: int

class BaseNode:
    def __init__(
        self, 
        children=None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        self.parents: Optional[List[BaseNode]] = None
        self.children: Optional[List[BaseNode]] = children
        self.output: Any = None
        self.model, self.using_native_model = initialize_model(model)

        # Set parent references for children
        if self.children is not None:
            for child in self.children:
                child.parents = []
                child.parents.append(self)

    def is_ready(self):
        """Check if all parent outputs have been computed"""
        return all(parent.output is not None for parent in self.parents)

    def run(self):
        """Run the DAG by processing nodes when they are ready (parent outputs computed)"""
        ready_nodes = [self]
        while ready_nodes:
            current_node = ready_nodes.pop(0)
            if current_node.is_ready():
                result = current_node.execute(self.model)
                if isinstance(result, StoppingClassificationScore):
                    return result
                ready_nodes.extend(current_node.children)
            else:
                ready_nodes.append(current_node)

    def execute(self):
        raise NotImplementedError("Subclasses must implement the `execute` method.")

######################################################
# Nodes ##############################################
######################################################

class ExtractionNode(BaseNode):
    def __init__(
        self, 
        task: str, 
        test_case: LLMTestCase, 
        test_case_params: List[LLMTestCaseParams],
        children: Optional[List['BaseNode']] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        super().__init__(children, model)
        self.task: str = task
        self.test_case: LLMTestCase = test_case
        self.test_case_params: List[LLMTestCaseParams] = test_case_params

    def generate(self) -> None:
        output_label_prompt = (self.task)
        output_prompt = (self.task, self.test_case, self.test_case_params)
        self.output_label = self.model.generate(output_label_prompt)
        self.output = self.model.generate(output_prompt)

class BinaryClassificationNode(BaseNode):
    def __init__(
        self, 
        criteria: str, 
        test_case: LLMTestCase, 
        test_case_params: List[LLMTestCaseParams],
        no_children: Union[BaseNode, StoppingClassificationScore],
        yes_children: Union[BaseNode, StoppingClassificationScore],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        super().__init__(model=model)
        self.criteria: str = criteria
        self.test_case: LLMTestCase = test_case
        self.test_case_params: List[LLMTestCaseParams] = test_case_params
        self.no_children = no_children
        self.yes_children = yes_children

    def get_parent_outputs(self):
        parent_outputs = []
        for parent in self.parents:
            if isinstance(parent, ExtractionNode):
                parent_outputs.append(f"""{parent.output_label}: {parent.output}""")
        return parent_outputs

    def generate(self) -> None:
        parent_outputs = self.get_parent_outputs()
        binary_classification_prompt = (self.criteria, self.test_case, self.test_case_params, parent_outputs)
        response: BinaryClassificationVerdict = self.model.generate(binary_classification_prompt, BinaryClassificationVerdict)
        verdict = response.verdict
        if verdict is True:
            return self.yes_children
        else:
            return self.no_children


class MulticlassClassificationNode(BaseNode):
    def __init__(
        self, 
        criteria: str, 
        test_case: LLMTestCase, 
        test_case_params: List[LLMTestCaseParams],
        children: List[Union[StoppingClassificationScore, BaseNode]],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        super().__init__(model=model)
        self.criteria: str = criteria
        self.test_case: LLMTestCase = test_case
        self.test_case_params: List[LLMTestCaseParams] = test_case_params

    def generate(self) -> None:
        output_label_prompt = (self.criteria)
        output_prompt = (self.criteria, self.test_case, self.test_case_params)
        self.output_label = self.model.generate(output_label_prompt)
        self.output = self.model.generate(output_prompt)
