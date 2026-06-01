from typing import Union, List, Dict
from pydantic import BaseModel


class GuidelineListSchema(BaseModel):
    guidelines: List[str]


class COPROProposalSchema(BaseModel):
    thought_process: str
    revised_prompt: Union[str, List[Dict[str, str]]]
