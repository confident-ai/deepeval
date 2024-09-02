"""Tests for various benchmark functions.
"""

# import pytest
# from deepeval.benchmarks import GSM8K
# from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
# from deepeval.dataset import Golden


# @pytest.skip
# def golden():
#     return Golden(
#         input="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
#         expected_output="72",
#     )


# @pytest.skip
# def train_sample():
#     return {
#         "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
#         "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
#     }


# def test_gsm8k_template(golden, train_sample):
#     max_n_shots = 3
#     train_set = [train_sample for _ in range(max_n_shots)]

#     no_cot_prompt = GSM8KTemplate.generate_output(
#         golden.input, train_set, n_shots=max_n_shots, enable_cot=False
#     )
#     assert (
#         "think step-by-step" not in no_cot_prompt
#     ), f"Did not expect to see chain-of-thought in the prompt {no_cot_prompt}"

#     cot_prompt = GSM8KTemplate.generate_output(
#         golden.input, train_set, n_shots=max_n_shots, enable_cot=True
#     )
#     assert (
#         "think step-by-step" in cot_prompt
#     ), f"Expected to see chain-of-thought in the prompt {cot_prompt}"

#     cot_prompt_zero_shot = GSM8KTemplate.generate_output(
#         golden.input, train_set, n_shots=max_n_shots, enable_cot=True
#     )
#     assert (
#         "think step-by-step" in cot_prompt_zero_shot
#     ), f"Expected to see chain-of-thought in the prompt {cot_prompt_zero_shot}"

########################################
from deepeval.benchmarks import *
from deepeval.benchmarks.tasks import *
from deepeval.benchmarks.modes import *
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.models.gpt_model import GPTModel

# gpt_model = SchematicGPTModel(model="gpt-4o")
gpt_model = GPTModel(model="gpt-4o")

benchmark_mmlu = MMLU(
    tasks=[MMLUTask.ABSTRACT_ALGEBRA, MMLUTask.MORAL_SCENARIOS]
)
benchmark_hellaswag = HellaSwag(tasks=[HellaSwagTask.APPLYING_SUNSCREEN])
benchmark_drop = DROP(tasks=[DROPTask.HISTORY_1450, DROPTask.NFL_899])
benchmark_truthfulQA = TruthfulQA(
    tasks=[TruthfulQATask.PSYCHOLOGY, TruthfulQATask.SUBJECTIVE]
)
benchmark_truthfulQA_MC2 = TruthfulQA(
    tasks=[TruthfulQATask.PSYCHOLOGY, TruthfulQATask.SUBJECTIVE],
    mode=TruthfulQAMode.MC2,
)

benchmark_gsm8k = GSM8K(n_problems=25)
benchmark_bbh = BigBenchHard(
    tasks=[
        BigBenchHardTask.BOOLEAN_EXPRESSIONS,
        BigBenchHardTask.CAUSAL_JUDGEMENT,
        BigBenchHardTask.DATE_UNDERSTANDING,
        BigBenchHardTask.DISAMBIGUATION_QA,
        BigBenchHardTask.DYCK_LANGUAGES,
        BigBenchHardTask.FORMAL_FALLACIES,
        BigBenchHardTask.GEOMETRIC_SHAPES,
        BigBenchHardTask.HYPERBATON,
        BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS,
        BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS,
        BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS,
        BigBenchHardTask.MOVIE_RECOMMENDATION,
        BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO,
        BigBenchHardTask.NAVIGATE,
        BigBenchHardTask.OBJECT_COUNTING,
        BigBenchHardTask.PENGUINS_IN_A_TABLE,
        BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS,
        BigBenchHardTask.RUIN_NAMES,
        BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION,
        BigBenchHardTask.SNARKS,
        BigBenchHardTask.SPORTS_UNDERSTANDING,
        BigBenchHardTask.TEMPORAL_SEQUENCES,
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS,
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS,
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS,
        BigBenchHardTask.WEB_OF_LIES,
        BigBenchHardTask.WORD_SORTING,
    ]
)

benchmarks = [
    # benchmark_mmlu,
    benchmark_hellaswag,
    benchmark_drop,
    benchmark_truthfulQA,
    benchmark_truthfulQA_MC2,
    benchmark_gsm8k,
    benchmark_bbh,  # Need to test every task (different schemas for all tasks)
]

# for benchmark in benchmarks:
#     benchmark.evaluate(model=gpt_model)
