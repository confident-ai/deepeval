from typing import Optional, Sequence, List, Union


def conform_contexts_type(
    contexts: Optional[Sequence[str]] = None,
) -> Union[List[str], None]:
    if contexts is None:
        return None

    return list(contexts)
