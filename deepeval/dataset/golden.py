import re
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Optional, Dict, List
from deepeval.test_case import ToolCall, Turn, MLLMImage
from deepeval.test_case.llm_test_case import _MLLM_IMAGE_REGISTRY


class Golden(BaseModel):
    input: str
    actual_output: Optional[str] = Field(
        default=None, serialization_alias="actualOutput"
    )
    expected_output: Optional[str] = Field(
        default=None, serialization_alias="expectedOutput"
    )
    context: Optional[List[str]] = Field(default=None)
    retrieval_context: Optional[List[str]] = Field(
        default=None, serialization_alias="retrievalContext"
    )
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    tools_called: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="expectedTools"
    )
    source_file: Optional[str] = Field(
        default=None, serialization_alias="sourceFile"
    )
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    multimodal: bool = Field(False, exclude=True)
    images_mapping: Dict[str, MLLMImage] = Field(
        default=None, alias="imagesMapping"
    )
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        auto_detect = (
            any(
                [
                    re.search(pattern, self.input or "") is not None,
                    re.search(pattern, self.actual_output or "") is not None,
                ]
            )
            if isinstance(self.input, str)
            else self.multimodal
        )
        if self.retrieval_context is not None:
            auto_detect = auto_detect or any(
                re.search(pattern, context) is not None
                for context in self.retrieval_context
            )
        if self.context is not None:
            auto_detect = auto_detect or any(
                re.search(pattern, context) is not None
                for context in self.context
            )

        self.multimodal = auto_detect

        return self

    def _get_images_mapping(self) -> Dict[str, MLLMImage]:
        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        image_ids = set()

        def extract_ids_from_string(s: Optional[str]) -> None:
            """Helper to extract image IDs from a string."""
            if s is not None and isinstance(s, str):
                matches = re.findall(pattern, s)
                image_ids.update(matches)

        def extract_ids_from_list(lst: Optional[List[str]]) -> None:
            """Helper to extract image IDs from a list of strings."""
            if lst is not None:
                for item in lst:
                    extract_ids_from_string(item)

        extract_ids_from_string(self.input)
        extract_ids_from_string(self.actual_output)
        extract_ids_from_string(self.expected_output)
        extract_ids_from_list(self.context)
        extract_ids_from_list(self.retrieval_context)

        images_mapping = {}
        for img_id in image_ids:
            if img_id in _MLLM_IMAGE_REGISTRY:
                images_mapping[img_id] = _MLLM_IMAGE_REGISTRY[img_id]

        return images_mapping if len(images_mapping) > 0 else None


class ConversationalGolden(BaseModel):
    scenario: str
    expected_outcome: Optional[str] = Field(
        None, serialization_alias="expectedOutcome"
    )
    user_description: Optional[str] = Field(
        None, serialization_alias="userDescription"
    )
    context: Optional[List[str]] = Field(default=None)
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    turns: Optional[List[Turn]] = Field(default=None)
    multimodal: bool = Field(False, exclude=True)
    images_mapping: Dict[str, MLLMImage] = Field(
        default=None, alias="imagesMapping"
    )
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        if self.scenario:
            if re.search(pattern, self.scenario) is not None:
                self.multimodal = True
                return self
        if self.expected_outcome:
            if re.search(pattern, self.expected_outcome) is not None:
                self.multimodal = True
                return self
        if self.user_description:
            if re.search(pattern, self.user_description) is not None:
                self.multimodal = True
                return self
        if self.turns:
            for turn in self.turns:
                if re.search(pattern, turn.content) is not None:
                    self.multimodal = True
                    return self
                if turn.retrieval_context is not None:
                    self.multimodal = any(
                        re.search(pattern, context) is not None
                        for context in turn.retrieval_context
                    )

        return self

    def _get_images_mapping(self) -> Dict[str, MLLMImage]:
        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        image_ids = set()

        def extract_ids_from_string(s: Optional[str]) -> None:
            """Helper to extract image IDs from a string."""
            if s is not None and isinstance(s, str):
                matches = re.findall(pattern, s)
                image_ids.update(matches)

        def extract_ids_from_list(lst: Optional[List[str]]) -> None:
            """Helper to extract image IDs from a list of strings."""
            if lst is not None:
                for item in lst:
                    extract_ids_from_string(item)

        extract_ids_from_string(self.scenario)
        extract_ids_from_string(self.expected_outcome)
        extract_ids_from_list(self.context)
        extract_ids_from_string(self.user_description)
        if self.turns:
            for turn in self.turns:
                extract_ids_from_string(turn.content)
                extract_ids_from_list(turn.retrieval_context)

        images_mapping = {}
        for img_id in image_ids:
            if img_id in _MLLM_IMAGE_REGISTRY:
                images_mapping[img_id] = _MLLM_IMAGE_REGISTRY[img_id]

        return images_mapping if len(images_mapping) > 0 else None
