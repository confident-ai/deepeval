from pydantic import BaseModel
from typing import Union, List, Dict


class DatasetSummarySchema(BaseModel):
    summary: str


class InstructionProposalSchema(BaseModel):
    thought_process: str
    revised_instruction: Union[str, List[Dict[str, str]]]
