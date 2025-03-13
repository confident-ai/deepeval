"""Tests for various benchmark functions."""

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
## Import ##############################
########################################

from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.models.gpt_model import GPTModel
from deepeval.benchmarks.tasks import *
from deepeval.benchmarks.modes import *
from deepeval.benchmarks import *
from typing import List

gpt_model = GPTModel(model="gpt-4o")
n_problems = 2
verbose_mode = True

########################################
## Benchmark ###########################
########################################

benchmark_mmlu = MMLU(
    tasks=[MMLUTask.ABSTRACT_ALGEBRA, MMLUTask.MORAL_SCENARIOS],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_hellaswag = HellaSwag(
    tasks=[HellaSwagTask.APPLYING_SUNSCREEN, HellaSwagTask.ARCHERY],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_bbh = BigBenchHard(
    tasks=[
        BigBenchHardTask.CAUSAL_JUDGEMENT,
        BigBenchHardTask.FORMAL_FALLACIES,
    ],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_drop = DROP(
    tasks=[DROPTask.HISTORY_1450, DROPTask.NFL_899],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_truthful_qa = TruthfulQA(
    tasks=[TruthfulQATask.PSYCHOLOGY, TruthfulQATask.SUBJECTIVE],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_human_eval = HumanEval(
    tasks=[HumanEvalTask.ADD_ELEMENTS, HumanEvalTask.CONCATENATE],
    n=3,
    verbose_mode=verbose_mode,
)
benchmark_squad = SQuAD(
    tasks=[SQuADTask.AMAZON_RAINFOREST, SQuADTask.APOLLO_PROGRAM],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_gsm8k = GSM8K(n_problems=25, verbose_mode=verbose_mode)
benchmark_mathqa = MathQA(
    tasks=[MathQATask.GENERAL, MathQATask.GEOMETRY],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_logiqa = LogiQA(
    tasks=[LogiQATask.CATEGORICAL_REASONING, LogiQATask.CONJUNCTIVE_REASONING],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_boolq = BoolQ(n_problems=n_problems, verbose_mode=verbose_mode)
benchmark_arc = ARC(
    n_problems=n_problems, mode=ARCMode.CHALLENGE, verbose_mode=verbose_mode
)
benchmark_bbq = BBQ(
    tasks=[BBQTask.DISABILITY_STATUS, BBQTask.NATIONALITY],
    n_problems_per_task=n_problems,
    verbose_mode=verbose_mode,
)
benchmark_lambada = LAMBADA(n_problems=n_problems, verbose_mode=verbose_mode)
benchmark_winogrande = Winogrande(
    n_problems=n_problems, verbose_mode=verbose_mode
)

########################################
## Evaluate ############################
########################################

benchmarks: List[DeepEvalBaseBenchmark] = [
    benchmark_mmlu,
    benchmark_hellaswag,
    benchmark_bbh,
    benchmark_drop,
    benchmark_truthful_qa,
    benchmark_human_eval,
    benchmark_squad,
    benchmark_gsm8k,
    benchmark_mathqa,
    benchmark_logiqa,
    benchmark_boolq,
    benchmark_arc,
    benchmark_bbq,
    benchmark_lambada,
    benchmark_winogrande,
]

for benchmark in benchmarks:
    try:
        benchmark.evaluate(model=gpt_model, k=2)
    except TypeError:
        benchmark.evaluate(model=gpt_model)
    print(benchmark.predictions)
