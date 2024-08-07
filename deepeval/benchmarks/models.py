from pydantic import BaseModel
from typing import List, Literal, Union


class MultipleChoiceModel(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class ListOfNumbersModel(BaseModel):
    answer: List[int]


class ListofStringsModel(BaseModel):
    answer: List[str]


class NumberModel(BaseModel):
    answer: int


class StringModel(BaseModel):
    answer: str


# DROP Models #############################


class DROPStringModel(BaseModel):
    answer: str


class DROPNumberModel(BaseModel):
    answer: int


class DROPDateModel(BaseModel):
    answer: str


# BBH Models #############################


class AffirmationModel(BaseModel):
    answer: Literal["No", "Yes"]


class AffirmationLowerModel(BaseModel):
    answer: Literal["no", "yes"]


class BooleanModel(BaseModel):
    answer: Literal["True", "False"]


class ValidModel(BaseModel):
    answer: Literal["valid", "invalid"]


class BBHMultipleChoice2(BaseModel):
    answer: Literal["(A)", "(B)"]


class BBHMultipleChoice3(BaseModel):
    answer: Literal["(A)", "(B)", "(C)"]


class BBHMultipleChoice4(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)"]


class BBHMultipleChoice5(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)"]


class BBHMultipleChoice6(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]


class BBHMultipleChoice7(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]


class BBHMultipleChoice11(BaseModel):
    answer: Literal[
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
        "(I)",
        "(J)",
        "(K)",
    ]


class BBHMultipleChoice18(BaseModel):
    answer: Literal[
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
        "(I)",
        "(J)",
        "(K)",
        "(L)",
        "(M)",
        "(N)",
        "(O)",
        "(P)",
        "(Q)",
        "(R)",
    ]


bbh_models_dict = {
    "boolean_expressions": BooleanModel,
    "causal_judgement": AffirmationModel,
    "date_understanding": BBHMultipleChoice6,
    "disambiguation_qa": BBHMultipleChoice3,
    "dyck_languages": StringModel,
    "formal_fallacies": ValidModel,
    "geometric_shapes": BBHMultipleChoice11,
    "hyperbaton": BBHMultipleChoice2,
    "logical_deduction_three_objects": BBHMultipleChoice3,
    "logical_deduction_five_objects": BBHMultipleChoice5,
    "logical_deduction_seven_objects": BBHMultipleChoice7,
    "movie_recommendation": BBHMultipleChoice5,
    "multistep_arithmetic_two": NumberModel,
    "navigate": AffirmationModel,
    "object_counting": NumberModel,
    "penguins_in_a_table": BBHMultipleChoice5,
    "reasoning_about_colored_objects": BBHMultipleChoice18,
    "ruin_names": BBHMultipleChoice4,
    "salient_translation_error_detection": BBHMultipleChoice6,
    "snarks": BBHMultipleChoice2,
    "sports_understanding": AffirmationLowerModel,
    "temporal_sequences": BBHMultipleChoice4,
    "tracking_shuffled_objects_three_objects": BBHMultipleChoice3,
    "tracking_shuffled_objects_five_objects": BBHMultipleChoice5,
    "tracking_shuffled_objects_seven_objects": BBHMultipleChoice7,
    "web_of_lies": AffirmationModel,
    "word_sorting": StringModel,
}

bbh_confinement_statements_dict = {
    "boolean_expression": "\n\nOutput 'True' or 'False'. Full answer not needed.",
    "causal_judgement": "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    "date_understanding": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    "disambiguation_qa": "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    "dyck_language": "\n\nOutput only the sequence of parenthases characters separated by white space. Full answer not needed.",
    "formal_fallacies": "\n\nOutput 'invalid' or 'valid'. Full answer not needed.",
    "geometric_shapes": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', or '(K)'. Full answer not needed.",
    "hyperbaton": "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    "logical_deduction_three_objects": "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    "logical_deduction_five_objects": "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    "logical_deduction_seven_objects": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    "movie_recommendation": "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    "multi_step_arithmetic": "\n\nOutput the numerical answer. Full answer not needed.",
    "navigate": "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    "object_counting": "\n\nOutput the numerical answer. Full answer not needed.",
    "penguins_in_a_table": "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    "reasoning_about_colored_objects": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', or '(R)'. Full answer not needed.",
    "ruin_names": "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    "salient_translation_error_detection": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    "snarks": "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    "sports_understanding": "\n\nOutput 'yes' or 'no'. Full answer not needed.",
    "temporal_sequences": "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    "tracking_shuffled_objects_three_objects": "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    "tracking_shuffled_objects_five_objects": "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    "tracking_shuffled_objects_seven_objects": "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    "web_of_lies": "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    "word_sorting": "\n\nOutput only the sequence of words separated by white space. Full answer not needed.",
}
