import asyncio
import importlib.util
import inspect
import yaml
import os
import sys
import glob
import json
import typer
import requests

import deepeval.metrics as metrics_module

from typing import Any, List, Union

from . import config
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.dataset import ConversationalGolden, EvaluationDataset, Golden
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.metrics import (
    BaseArenaMetric,
    BaseConversationalMetric,
    BaseMetric,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.simulator import ConversationSimulator
from deepeval.utils import get_or_create_event_loop


def execute_cicd(config_file: str, ci_provider: str = "github"):
    """
    Run CI/CD evaluation from a YAML config (dataset, model_callback, metrics).
    """

    try:
        cfg = _load_config(config_file)
    except Exception as e:
        print(f"Failed to load config file '{config_file}': {e}")
        sys.exit(1)

    config.apply_env()

    raw_model_callback = cfg.get("model_callback", {})
    try:
        model_callback = _load_model_callback_from_file(
            raw_model_callback.get("file_path"),
            raw_model_callback.get("function_name"),
        )
    except Exception as e:
        print(f"Failed to load model callback: {e}")
        sys.exit(1)

    dataset = cfg.get("dataset", {})
    try:
        goldens = _build_goldens_from_dataset(dataset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    if len(goldens) == 0:
        print("Dataset is empty, please add goldens to the dataset")
        sys.exit(1)

    try:
        metrics = _build_metrics(cfg.get("metrics", []))
    except Exception as e:
        print(f"Failed to build metrics: {e}")
        sys.exit(1)

    if len(metrics) == 0:
        print("No metrics provided, please add metrics to the config")
        sys.exit(1)

    try:
        test_cases = _create_test_cases_from_goldens(goldens, model_callback)
    except Exception as e:
        print(f"Failed to create test cases from goldens: {e}")
        sys.exit(1)

    try:
        results = evaluate(
            test_cases,
            metrics,
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
        )
    except Exception as e:
        error_md = f"### ⚠️ DeepEval Run Failed\nAn internal error occurred during evaluation:\n```\n{str(e)}\n```"
        _post_ci_comment(error_md, ci_provider)
        sys.exit(1)

    try:
        raw_required_metrics = cfg.get("required_metrics", [])
        if not isinstance(raw_required_metrics, list):
            raise ValueError("'required_metrics' must be a list.")

        required_metric_identifiers = [
            metric.get("name")
            for metric in raw_required_metrics
            if isinstance(metric, dict) and metric.get("name")
        ]

        required_metrics = []
        for identifier in required_metric_identifiers:
            matched = False
            for m in metrics:
                if identifier in [
                    m.__class__.__name__,
                    getattr(m, "__name__", None),
                    getattr(m, "name", None),
                ]:
                    required_metrics.append(m.__name__)
                    matched = True
            if not matched:
                required_metrics.append(identifier)

    except Exception as e:
        print(f"Failed to parse required metrics: {e}")
        sys.exit(1)

    try:
        pass_rate = cfg.get("pass_rate", None)
        if pass_rate is not None and not isinstance(pass_rate, (int, float)):
            raise ValueError("'pass_rate' must be a number.")
    except Exception as e:
        print(f"Failed to parse pass rate: {e}")
        sys.exit(1)

    test_results = results.test_results
    passed = True
    for test_result in test_results:
        metrics_data = test_result.metrics_data
        for metric_data in metrics_data:
            if metric_data.name in required_metrics:
                if not metric_data.success:
                    passed = False
                    break
    test_run_pass_rate = len(
        [test_result for test_result in test_results if test_result.success]
    ) / len(test_results)
    if pass_rate and test_run_pass_rate < pass_rate:
        passed = False

    report = EvaluationConsoleReport(results.test_results)
    output_dir = ".deepeval_ci_reports"

    try:
        report.export_to_cicd_markdown(
            output_dir=output_dir, evaluation_name="pr_evaluation"
        )

        # 2. Find the generated markdown file
        list_of_files = glob.glob(f"{output_dir}/*.md")
        if not list_of_files:
            raise FileNotFoundError(
                f"No markdown reports found in {output_dir}"
            )
        latest_report_path = max(list_of_files, key=os.path.getctime)

        with open(latest_report_path, "r", encoding="utf-8") as f:
            markdown_summary = f.read()
    except Exception as e:
        print(f"Failed to generate or read evaluation report: {e}")
        sys.exit(1)

    # Inject the Confident AI Link if it exists
    if results.confident_link:
        markdown_summary += (
            f"\n### 🔍 Deep Dive\n"
            f"[**View the full results on Confident AI**]({results.confident_link})"
        )
    else:
        markdown_summary += f"\nSet CONFIDENT_API_KEY to view these results on the Confident AI platform"

    # 3. Try to post it to the CI provider
    _post_ci_comment(markdown_summary, ci_provider)

    if not passed:
        print("Test run failed")
        sys.exit(1)
    else:
        print("Test run passed")
        sys.exit(0)


########################################################
###### Utils ###########################################
########################################################


def _load_config(path: str):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not read or parse YAML file at '{path}': {e}"
        )


def _load_model_callback_from_file(
    file_path: str, function_name: str = "model_callback"
) -> Any:
    """Load a model callback function from a Python file."""
    if not file_path:
        raise ValueError("file_path for model_callback is not provided.")
    if not function_name:
        function_name = "model_callback"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model callback file not found: {file_path}")

    try:
        spec = importlib.util.spec_from_file_location(
            "deepeval_cicd_model_callback_module", file_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["deepeval_cicd_model_callback_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to execute module from {file_path}: {e}")

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}"
        )

    callback = getattr(module, function_name)
    if not callable(callback):
        raise TypeError(f"'{function_name}' in {file_path} is not callable")

    return callback


def _build_goldens_from_dataset(
    raw_dataset,
) -> List[Union[Golden, ConversationalGolden]]:
    goldens: List[Union[Golden, ConversationalGolden]] = []

    if not isinstance(raw_dataset, dict):
        raise ValueError("Dataset configuration must be a dictionary.")

    if raw_dataset.get("alias"):
        try:
            dataset = EvaluationDataset()
            dataset.pull(alias=raw_dataset.get("alias"))
            goldens = dataset.goldens
        except Exception as e:
            raise RuntimeError(
                f"Failed to pull dataset with alias '{raw_dataset.get('alias')}': {e}"
            )

        if len(goldens) == 0:
            raise ValueError(
                f"Dataset with alias '{raw_dataset.get('alias')}' is empty. Please add goldens to the dataset."
            )

    elif raw_dataset.get("path"):
        try:
            dataset = EvaluationDataset()
            dataset.add_goldens_from_json_file(
                file_path=raw_dataset.get("path")
            )
            goldens = dataset.goldens
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from path '{raw_dataset.get('path')}': {e}"
            )

        if len(goldens) == 0:
            raise ValueError(
                f"Dataset at path '{raw_dataset.get('path')}' is empty. Please add goldens to the dataset."
            )

    elif raw_dataset.get("goldens"):
        raw_goldens = raw_dataset.get("goldens", [])
        if not isinstance(raw_goldens, list):
            raise ValueError("'goldens' must be a list of golden objects.")

        if len(raw_goldens) == 0:
            raise ValueError(
                "The 'goldens' list is empty. Please add goldens to the dataset."
            )

        try:
            if "input" in raw_goldens[0]:
                new_goldens = []
                for i, golden in enumerate(raw_goldens):
                    if not isinstance(golden, dict):
                        raise ValueError(
                            f"Golden at index {i} is not a dictionary."
                        )
                    if "input" not in golden:
                        raise ValueError(
                            f"Golden at index {i} is missing the required 'input' field for single-turn goldens. Mixed golden types are not allowed."
                        )
                    new_goldens.append(
                        Golden(
                            input=golden.get("input"),
                            expected_output=golden.get("expected_output"),
                            actual_output=golden.get("actual_output"),
                            context=golden.get("context"),
                            retrieval_context=golden.get("retrieval_context"),
                        )
                    )
                goldens.extend(new_goldens)
            else:
                new_goldens = []
                for i, golden in enumerate(raw_goldens):
                    if not isinstance(golden, dict):
                        raise ValueError(
                            f"Golden at index {i} is not a dictionary."
                        )
                    if "input" in golden:
                        raise ValueError(
                            f"Golden at index {i} contains 'input', which is not allowed for conversational goldens. Mixed golden types are not allowed."
                        )
                    new_goldens.append(
                        ConversationalGolden(
                            scenario=golden.get("scenario"),
                            expected_outcome=golden.get("expected_outcome"),
                            name=golden.get("name"),
                        )
                    )
                goldens.extend(new_goldens)
        except Exception as e:
            raise RuntimeError(f"Failed to parse 'goldens' from dataset: {e}")
    else:
        raise ValueError(
            "Dataset configuration must contain 'alias', 'path', or 'goldens'."
        )

    return goldens


def _build_metrics(
    raw_metrics: List[dict],
) -> List[Union[BaseMetric, BaseConversationalMetric, BaseArenaMetric]]:
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseArenaMetric]
    ] = []
    for i, item in enumerate(raw_metrics):
        if not isinstance(item, dict):
            raise TypeError(
                f"metrics[{i}] must be a dict, got {type(item).__name__}"
            )
        class_name = item.get("metric")
        if not class_name or not isinstance(class_name, str):
            raise ValueError(
                f"metrics[{i}] requires a non-empty string 'metric' (metric class name)"
            )
        cls = getattr(metrics_module, class_name, None)
        if cls is None:
            raise ValueError(
                f"Unknown metric '{class_name}'. Please use a valid metric from deepeval.metrics."
            )
        kwargs = {k: v for k, v in item.items() if k != "metric"}
        try:
            metrics.append(cls(**kwargs))
        except TypeError as e:
            raise TypeError(
                f"Failed to construct {class_name} with kwargs {kwargs!r}: {e}"
            ) from e
    return metrics


def _create_test_cases_from_goldens(
    goldens: List[Union[Golden, ConversationalGolden]], model_callback: Any
) -> List[Union[LLMTestCase, ConversationalTestCase]]:
    if not goldens:
        raise ValueError("Goldens list is empty.")

    if isinstance(goldens[0], ConversationalGolden):
        return _get_conversational_test_cases_from_goldens(
            goldens, model_callback
        )
    else:
        return _get_llm_test_cases_from_goldens(goldens, model_callback)


def _get_conversational_test_cases_from_goldens(
    goldens: List[ConversationalGolden],
    model_callback: Any,
) -> List[ConversationalTestCase]:
    try:
        simulator = ConversationSimulator(model_callback=model_callback)
        test_cases = simulator.simulate(conversational_goldens=goldens)
        return test_cases
    except Exception as e:
        raise RuntimeError(f"Failed to simulate conversational test cases: {e}")


def _get_llm_test_cases_from_goldens(
    goldens: List[Golden],
    model_callback: Any,
) -> List[LLMTestCase]:
    async def _run_all() -> List[LLMTestCase]:
        tasks = [
            _process_single_golden(golden, model_callback) for golden in goldens
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(_run_all())
    except Exception as e:
        raise RuntimeError(f"Failed to create LLM test cases: {e}")


async def _process_single_golden(
    golden: Golden, model_callback: Any
) -> LLMTestCase:
    try:
        if inspect.iscoroutinefunction(model_callback):
            raw_result = await model_callback(golden.input)
        else:
            loop = asyncio.get_running_loop()
            raw_result = await loop.run_in_executor(
                None, model_callback, golden.input
            )

        if isinstance(raw_result, Turn):
            turn = raw_result
        elif isinstance(raw_result, dict):
            data = dict(raw_result)
            data.setdefault("role", "assistant")
            turn = Turn.model_validate(data)
        else:
            raise TypeError(
                f"model_callback must return a Turn instance or a dict; got {type(raw_result).__name__}"
            )
        return LLMTestCase(
            input=golden.input,
            actual_output=turn.content,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=turn.retrieval_context,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error processing golden with input '{golden.input}': {e}"
        )


def _post_ci_comment(markdown_content: str, ci_provider: str):
    if ci_provider.lower() == "github":
        _post_github_pr_comment(markdown_content)
    else:
        print(
            f"⚠️ CI provider '{ci_provider}' is not yet supported for automated PR comments."
        )


def _post_github_pr_comment(markdown_content: str):
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # format: "owner/repo"
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not token or not repo or not event_path:
        missing = [
            name
            for name, val in (
                ("GITHUB_TOKEN", token),
                ("GITHUB_REPOSITORY", repo),
                ("GITHUB_EVENT_PATH", event_path),
            )
            if not val
        ]
        print(
            "Skipping PR comment: missing environment variable(s): "
            + ", ".join(missing)
            + ". (GitHub Actions sets these automatically; local runs skip posting.)"
        )
        return

    # Extract the PR number from the event payload
    try:
        with open(event_path, "r") as f:
            event_data = json.load(f)
            if "pull_request" not in event_data:
                print("Event is not a pull request. Skipping PR comment.")
                return
            pr_number = event_data["pull_request"]["number"]
    except Exception as e:
        print(f"Failed to read GitHub event payload: {e}")
        return

    base_url = (
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        # 1. Fetch existing comments
        response = requests.get(base_url, headers=headers)
        if response.status_code != 200:
            print(
                f"⚠️ Failed to fetch existing comments from GitHub. Status: {response.status_code}, Response: {response.text}"
            )
            return

        existing_comments = response.json()

        if not isinstance(existing_comments, list):
            print(
                f"⚠️ Unexpected response format from GitHub API when fetching comments. Expected a list, got: {type(existing_comments).__name__}. Response: {existing_comments}"
            )
            return

        # 2. Look for our specific bot comment
        comment_id_to_update = None
        for comment in existing_comments:
            if not isinstance(comment, dict):
                continue
            if "🚀 DeepEval Evaluation Results" in comment.get("body", ""):
                comment_id_to_update = comment["id"]
                break

        # 3. Update or Create
        if comment_id_to_update:
            patch_url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id_to_update}"
            response = requests.patch(
                patch_url, json={"body": markdown_content}, headers=headers
            )
            action = "updated"
        else:
            response = requests.post(
                base_url, json={"body": markdown_content}, headers=headers
            )
            action = "posted"

        if response.status_code in [200, 201]:
            print(f"✅ Successfully {action} DeepEval results on GitHub PR.")
        else:
            print(
                f"⚠️ Failed to {action} PR comment. Status: {response.status_code}, Response: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error while posting to GitHub: {e}")
