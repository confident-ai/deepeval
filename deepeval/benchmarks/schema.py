from pydantic import BaseModel
from typing import List, Literal


class MultipleChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class ListOfNumbersSchema(BaseModel):
    answer: List[int]


class ListofStringsSchema(BaseModel):
    answer: List[str]


class NumberSchema(BaseModel):
    answer: int


class StringSchema(BaseModel):
    answer: str


# Winogrande Models #############################


class BinaryChoiceSchema(BaseModel):
    answer: Literal["A", "B"]


# BBQ Models #############################


class TrinaryChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C"]


# MathQA Models #############################


class MultipleChoiceSchemaLower(BaseModel):
    answer: Literal["a", "b", "c", "d"]


# DROP Models #############################


class DROPStringSchema(BaseModel):
    answer: str


class DROPNumberSchema(BaseModel):
    answer: int


class DROPDateSchema(BaseModel):
    answer: str


# BBH Models #############################


class AffirmationSchema(BaseModel):
    answer: Literal["No", "Yes"]


class AffirmationLowerSchema(BaseModel):
    answer: Literal["no", "yes"]


class BooleanSchema(BaseModel):
    answer: Literal["True", "False"]


class ValidSchema(BaseModel):
    answer: Literal["valid", "invalid"]


class BBHMultipleChoice2Schema(BaseModel):
    answer: Literal["(A)", "(B)"]


class BBHMultipleChoice3Schema(BaseModel):
    answer: Literal["(A)", "(B)", "(C)"]


class BBHMultipleChoice4Schema(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)"]


class BBHMultipleChoice5Schema(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)"]


class BBHMultipleChoice6Schema(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]


class BBHMultipleChoice7Schema(BaseModel):
    answer: Literal["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]


class BBHMultipleChoice11Schema(BaseModel):
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


class BBHMultipleChoice18Schema(BaseModel):
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
    "boolean_expressions": BooleanSchema,
    "causal_judgement": AffirmationSchema,
    "date_understanding": BBHMultipleChoice6Schema,
    "disambiguation_qa": BBHMultipleChoice3Schema,
    "dyck_languages": StringSchema,
    "formal_fallacies": ValidSchema,
    "geometric_shapes": BBHMultipleChoice11Schema,
    "hyperbaton": BBHMultipleChoice2Schema,
    "logical_deduction_three_objects": BBHMultipleChoice3Schema,
    "logical_deduction_five_objects": BBHMultipleChoice5Schema,
    "logical_deduction_seven_objects": BBHMultipleChoice7Schema,
    "movie_recommendation": BBHMultipleChoice5Schema,
    "multistep_arithmetic_two": NumberSchema,
    "navigate": AffirmationSchema,
    "object_counting": NumberSchema,
    "penguins_in_a_table": BBHMultipleChoice5Schema,
    "reasoning_about_colored_objects": BBHMultipleChoice18Schema,
    "ruin_names": BBHMultipleChoice4Schema,
    "salient_translation_error_detection": BBHMultipleChoice6Schema,
    "snarks": BBHMultipleChoice2Schema,
    "sports_understanding": AffirmationLowerSchema,
    "temporal_sequences": BBHMultipleChoice4Schema,
    "tracking_shuffled_objects_three_objects": BBHMultipleChoice3Schema,
    "tracking_shuffled_objects_five_objects": BBHMultipleChoice5Schema,
    "tracking_shuffled_objects_seven_objects": BBHMultipleChoice7Schema,
    "web_of_lies": AffirmationSchema,
    "word_sorting": StringSchema,
}
