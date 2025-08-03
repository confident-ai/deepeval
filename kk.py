# from deepeval.test_case import Turn
# from deepeval.metrics.utils import get_unit_interactions


# def make_turns(seq):
#     """Helper to create Turn objects from list of (role, content)."""
#     return [Turn(role, content) for role, content in seq]


# seq = [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
# expected = [[("user", "u1"), ("assistant", "a1")]]  # last user ignored
# result = get_unit_interactions(make_turns(seq))

# print(result)
