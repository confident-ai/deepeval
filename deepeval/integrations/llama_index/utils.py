from typing import List, Optional, Sequence, Union


def conform_contexts_type(
    contexts: Optional[Sequence[str]] = None,
) -> Union[List[str], None]:
    if contexts is None:
        return None

    return list(contexts)
