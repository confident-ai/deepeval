from pydantic import Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from PIL.Image import Image as ImageType

class MLLMTestCaseParams(Enum):
    INPUT_TEXT = "input_text"
    ACTUAL_OUTPUT_IMAGE = "actual_output_image"
    INPUT_IMAGE = "input_image"
    ACTUAL_OUTPUT_TEXT = "actual_output_text"

@dataclass
class MLLMTestCase:
    input_text: str
    actual_output_image: ImageType
    input_image: Optional[ImageType] = None
    actual_output_text: Optional[str] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Ensure `image_input` is None or an ImageType
        if self.input_image is not None:
            if not isinstance(self.input_image, ImageType):
                raise TypeError("'input_image' must be None or a PIL Image")

        # Ensure `actual_output_text` is None or a string
        if self.actual_output_text is not None:
            if not isinstance(self.actual_output_text, str):
                raise TypeError(
                    "'actual_output_text' must be None or a string"
                )