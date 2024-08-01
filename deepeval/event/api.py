from pydantic import BaseModel


class Link(BaseModel):
    value: str
