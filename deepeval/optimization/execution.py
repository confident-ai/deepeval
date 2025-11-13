from __future__ import annotations
from typing import Dict, Union, Protocol, TYPE_CHECKING
from deepeval.prompt.prompt import Prompt
from deepeval.optimization.types import ModuleId

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class PromptExecutor(Protocol):
    """
    Executes the user's observed app/agent with the provided prompts on a single golden
    and returns the final textual output to be scored by metrics.

    Implementers may provide either:
      - run (sync)
      - a_run (async)
    Providing both is fine, but at least one is required.
    """

    def run(
        self,
        prompts_by_module: Dict[ModuleId, Prompt],
        golden: Union[Golden, ConversationalGolden],
    ) -> str: ...

    async def a_run(
        self,
        prompts_by_module: Dict[ModuleId, Prompt],
        golden: Union[Golden, ConversationalGolden],
    ) -> str: ...
