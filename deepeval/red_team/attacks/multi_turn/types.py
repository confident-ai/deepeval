from typing import Callable, List, Optional

from deepteam.test_case.test_case import RTTurn

CallbackType = Callable[[str, Optional[List[RTTurn]]], RTTurn]
