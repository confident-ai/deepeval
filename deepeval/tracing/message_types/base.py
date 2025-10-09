from typing import Literal
from pydantic import BaseModel

class BaseMessage(BaseModel):
    role: Literal["user", "assistant"]