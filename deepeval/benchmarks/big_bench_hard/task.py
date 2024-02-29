from enum import Enum


class BigBenchHardTask(Enum):
    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    CAUSAL_JUDGEMENT = "causal_judgement"
    DATE_UNDERSTANDING = "date_understanding"
    DISAMBIGUATION_QA = "disambiguation_qa"
    DYCK_LANGUAGES = "dyck_languages"
    FORMAL_FALLACIES = "formal_fallacies"
    GEOMETRIC_SHAPES = "geometric_shapes"
    HYPERBATON = "hyperbaton"
    LOGICAL_DEDUCTION_FIVE_OBJECTS = "logical_deduction_five_objects"
    LOGICAL_DEDUCTION_SEVEN_OBJECTS = "logical_deduction_seven_objects"
    LOGICAL_DEDUCTION_THREE_OBJECTS = "logical_deduction_three_objects"
    MOVIE_RECOMMENDATION = "movie_recommendation"
    MULTISTEP_ARITHMETIC_TWO = "multistep_arithmetic_two"
    NAVIGATE = "navigate"
    OBJECT_COUNTING = "object_counting"
    PENGUINS_IN_A_TABLE = "penguins_in_a_table"
    REASONING_ABOUT_COLORED_OBJECTS = "reasoning_about_colored_objects"
    RUIN_NAMES = "ruin_names"
    SALIENT_TRANSLATION_ERROR_DETECTION = "salient_translation_error_detection"
    SNARKS = "snarks"
    SPORTS_UNDERSTANDING = "sports_understanding"
    TEMPORAL_SEQUENCES = "temporal_sequences"
    TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS = (
        "tracking_shuffled_objects_five_objects"
    )
    TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS = (
        "tracking_shuffled_objects_seven_objects"
    )
    TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS = (
        "tracking_shuffled_objects_three_objects"
    )
    WEB_OF_LIES = "web_of_lies"
    WORD_SORTING = "word_sorting"
