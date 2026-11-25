import asyncio
import importlib.util
from typing import Optional

from ..schema import CodeFindingsList
from .base import OUTPUT_FORMAT_INSTRUCTION, extract_findings, missing_sdk


class ClaudeAgentEngine:
    """
    Delegates scanning to the Claude Agent SDK (uses ANTHROPIC_API_KEY).
    """

    def __init__(self, model: Optional[str] = None):
        if importlib.util.find_spec("claude_agent_sdk") is None:
            raise missing_sdk("claude-code")
        self.model = model

    async def a_generate_findings(self, prompt: str) -> CodeFindingsList:
        from claude_agent_sdk import query, ClaudeAgentOptions

        options_kwargs = {"allowed_tools": []}
        if self.model:
            options_kwargs["model"] = self.model
        options = ClaudeAgentOptions(**options_kwargs)

        text = ""
        async for message in query(
            prompt=prompt + OUTPUT_FORMAT_INSTRUCTION, options=options
        ):
            result = getattr(message, "result", None)
            if isinstance(result, str):
                text = result
        return extract_findings(text)

    def generate_findings(self, prompt: str) -> CodeFindingsList:
        return asyncio.run(self.a_generate_findings(prompt))
