from pydantic import BaseModel
from typing import Union, List, Dict


class RewriterSchema(BaseModel):
    thought_process: str
    revised_prompt: Union[str, List[Dict[str, str]]]
