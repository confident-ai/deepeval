from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn


def dump_conversational_golden(golden: ConversationalGolden):
    new_golden = ConversationalGolden(
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        user_description=golden.user_description,
        context=golden.context,
        turns=(
            [
                Turn(
                    role=turn.role,
                    content=turn.content,
                    user_id=turn.user_id,
                    retrieval_context=turn.retrieval_context,
                    tools_called=turn.tools_called,
                )
                for turn in golden.turns
            ]
            if golden.turns is not None
            else None
        ),
    )
    try:
        body = new_golden.model_dump(
            by_alias=True,
            exclude_none=True,
            exclude={"turns": {"__all__": {"_mcp_interaction"}}},
        )
    except AttributeError:
        body = new_golden.dict(
            by_alias=True,
            exclude_none=True,
            exclude={"turns": {"__all__": {"_mcp_interaction"}}},
        )
    return {"conversationalGolden": body}
