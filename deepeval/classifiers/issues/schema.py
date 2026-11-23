from pydantic import BaseModel, Field


class IssueReview(BaseModel):
    agent_behavior: str = ""
    harm_done: str = ""
    agent_instruction: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str = ""

    def explanation(self) -> str:
        agent_instruction = self.agent_instruction.strip().rstrip(".")
        if not agent_instruction:
            return self.reason
        return " ".join(
            part
            for part in (
                f'The agent was told to do this: "{agent_instruction}".',
                self.reason,
            )
            if part
        )
