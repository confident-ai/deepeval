from typing import Callable, List, Optional

from deepeval.red_team.test_case.test_case import RTTurn

CallbackType = Callable[[str, Optional[List[RTTurn]]], RTTurn]
