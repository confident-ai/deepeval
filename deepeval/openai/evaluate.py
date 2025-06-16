from ast import keyword
import asyncio
import atexit
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.openai.extractors import InputParameters
from deepeval.test_run import auto_log_hyperparameters
from deepeval.evaluate import AsyncConfig

@dataclass
class TestCaseMetricPair:
    test_case: LLMTestCase
    metrics: List[BaseMetric]

@dataclass
class TestCasesMetricSet:
    test_cases: List[LLMTestCase]
    metrics: List[BaseMetric]

test_case_pairs: List[TestCaseMetricPair] = []


def add_test_case(test_case: LLMTestCase, metrics: List[BaseMetric]):
    test_case_pairs.append(TestCaseMetricPair(test_case=test_case, metrics=metrics))

##############################################
# Evaluation
##############################################

async def evaluate_async():
    if not test_case_pairs:
        return
    grouped: Dict[str, TestCasesMetricSet] = {}
    for pair in test_case_pairs:
        if pair.metrics:
            key = "".join([metric.__name__ for metric in pair.metrics])
            if key not in grouped:
                grouped[key] = TestCasesMetricSet(
                    test_cases=[pair.test_case], 
                    metrics=pair.metrics
                )
            else:
                grouped[key].test_cases.append(pair.test_case)
    for key, cases in grouped.items():
        evaluate(test_cases=cases.test_cases, metrics=cases.metrics)


def evaluate_sync():
    sync_config = AsyncConfig(run_async=False)
    if not test_case_pairs:
        return
    grouped: Dict[str, TestCasesMetricSet] = {}
    for pair in test_case_pairs:
        if pair.metrics:
            key = "".join([metric.__name__ for metric in pair.metrics])
            if key not in grouped:
                grouped[key] = TestCasesMetricSet(
                    test_cases=[pair.test_case], 
                    metrics=pair.metrics
                )
            else:
                grouped[key].test_cases.append(pair.test_case)
    for key, cases in grouped.items():
        evaluate(test_cases=cases.test_cases, metrics=cases.metrics, async_config=sync_config)

@atexit.register
def run_evaluations_atexit():
    if test_case_pairs:
        try:
            loop = asyncio.get_event_loop()
            loop_is_running =  loop.is_running()
            if loop_is_running:
                loop.create_task(evaluate_async())
            else:
                evaluate_sync()
        except Exception as e:
            print("⚠️ Could not schedule async evaluation in atexit: ", e)

##############################################
# Hyperparameters
##############################################

def log_hyperparameters(input_parameters: InputParameters):
    hyperparameters = {"model": input_parameters.model}
    if input_parameters.instructions:
        hyperparameters["system_prompt"] = input_parameters.instructions
    elif input_parameters.messages:
        system_messages = [m["content"] for m in input_parameters.messages if m["role"] == "system"]
        if system_messages:
            hyperparameters["system_prompt"] = (
                system_messages[0] if len(system_messages) == 1 else str(system_messages)
            )
    auto_log_hyperparameters(hyperparameters)
