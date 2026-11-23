from pydantic import BaseModel, Field


class IssueReview(BaseModel):
    what_happened: str = ""
    harm: str = ""
    instruction: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str = ""

    def explanation(self) -> str:
        instruction = self.instruction.strip().rstrip(".")
        if not instruction:
            return self.reason
        return " ".join(
            part
            for part in (
                f'The agent was told to do this: "{instruction}".',
                self.reason,
            )
            if part
        )
