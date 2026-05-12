from typing import Union, List, Dict
from pydantic import BaseModel


class SIMBARewriteSchema(BaseModel):
    discussion: str
    revised_prompt: Union[str, List[Dict[str, str]]]
