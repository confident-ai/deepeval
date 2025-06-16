import asyncio
import atexit
from dataclasses import dataclass
from typing import List, Optional
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

test_case_pairs: List[TestCaseMetricPair] = []

##############################################
# Test Case Registration
##############################################

def add_test_case(test_case: LLMTestCase, metrics: List[BaseMetric]):
    test_case_pairs.append(TestCaseMetricPair(test_case=test_case, metrics=metrics))

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


##############################################
# Async Evaluation Function
##############################################

async def evaluate_async():
    if not test_case_pairs:
        return
    grouped = defaultdict(list)
    for pair in test_case_pairs:
        if pair.metrics:
            grouped[frozenset(pair.metrics)].append(pair.test_case)
    for metric_set, cases in grouped.items():
        evaluate(test_cases=cases, metrics=list(metric_set))

def evaluate_sync():
    sync_config = AsyncConfig(run_async=False)
    if not test_case_pairs:
        return
    grouped = defaultdict(list)
    for pair in test_case_pairs:
        if pair.metrics:
            key_list = []
            print(pair.metrics)
            for metric in pair.metrics:
                key_list.append(metric.__class__)
            print(key_list)
            print("======\n"*10)
            key = str(key_list.sort())
            grouped[key].append(pair.test_case)
    for metric_set, cases in grouped.items():
        evaluate(test_cases=cases, metrics=list(metric_set), async_config=sync_config)


##############################################
# Fallback Atexit (non-blocking best-effort)
##############################################

@atexit.register
def _schedule_if_loop_is_alive():
    try:
        loop = asyncio.get_event_loop()
        loop_is_running =  loop.is_running()
        if loop_is_running:
            loop.create_task(evaluate_async())
        else:
            evaluate_sync()
    except Exception as e:
        print(e)
        print("⚠️ Could not schedule async evaluation in atexit.")
