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
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.metrics import BaseArenaMetric, BaseConversationalMetric, BaseMetric
from deepeval.test_case import LLMTestCase, Turn
from deepeval.utils import get_or_create_event_loop


app = typer.Typer(name="cicd", invoke_without_command=True)


@app.callback()
def cicd(
    config_file: str = typer.Argument(
        ...,
        help="Path to the CICD YAML config file.",
    ),
):
    """
    Run CI/CD evaluation from a YAML config (dataset, model_callback, metrics).
    """
    
    cfg = _load_config(config_file)
    config.apply_env()
    
    raw_model_callback = cfg.get("model_callback", {})
    model_callback = _load_model_callback_from_file(
        raw_model_callback.get("file_path"), 
        raw_model_callback.get("function_name")
    )

    dataset = cfg.get("dataset", {})

    if dataset.get("alias"):
        dataset = EvaluationDataset()
        dataset.pull(alias=dataset.get("alias"))
        goldens = dataset.goldens
        if len(goldens) == 0:
            print("Dataset is empty, please add goldens to the dataset")
            sys.exit(1) # Prolly need to log this as warning and not raise error here for sys exit
    elif dataset.get("path"):
        dataset = EvaluationDataset()
        dataset.add_goldens_from_json_file(file_path=dataset.get("path"))
        goldens = dataset.goldens
        if len(goldens) == 0:
            print("Dataset is empty, please add goldens to the dataset")
            sys.exit(1)
    elif dataset.get("goldens"):
        raw_goldens = dataset.get("goldens", [])
        goldens = []
        for golden in raw_goldens:
            new_golden = Golden(
                input=golden.get("input"),
                expected_output=golden.get("expected_output"),
                actual_output=golden.get("actual_output"),
                context=golden.get("context"),
                retrieval_context=golden.get("retrieval_context"),
                tools_called=golden.get("tools_called"),
                expected_tools=golden.get("expected_tools"),
            )
            goldens.append(new_golden)
        if len(goldens) == 0:
            print("Dataset is empty, please add goldens to the dataset")
            sys.exit(1)
    else: # TODO: Need to add suoport for conversational datasets
        raise ValueError("Dataset is not configured correctly")

    metrics = _build_metrics_from_config(cfg.get("metrics"))

    test_cases = _create_test_cases_from_goldens(goldens, model_callback)


    try:
        results = evaluate(test_cases, metrics, display_config=DisplayConfig(
            show_indicator=False, print_results=False
        ))
    except Exception as e:
        error_md = f"### ⚠️ DeepEval Run Failed\nAn internal error occurred during evaluation:\n```\n{str(e)}\n```"
        _post_github_pr_comment(error_md)
        sys.exit(1)

    required_metrics = cfg.get("required_metrics", {})
    required_metrics = [metric.get("name") for metric in required_metrics]

    pass_rate = cfg.get("pass_rate", None)

    test_results = results.test_results
    passed = True
    for test_result in test_results:
        metrics_data = test_result.metrics_data
        for metric_data in metrics_data:
            if metric_data.name in required_metrics:
                if not metric_data.success:
                    passed = False
                    break
    test_run_pass_rate = len([test_result for test_result in test_results if test_result.success]) / len(test_results)
    if pass_rate and test_run_pass_rate < pass_rate:
        passed = False

    report = EvaluationConsoleReport(results.test_results)
    output_dir = ".deepeval_ci_reports"
    report.export_to_markdown(output_dir=output_dir, evaluation_name="pr_evaluation")

    # 2. Find the generated markdown file
    list_of_files = glob.glob(f'{output_dir}/*.md')
    latest_report_path = max(list_of_files, key=os.path.getctime)
    
    with open(latest_report_path, "r", encoding="utf-8") as f:
        markdown_summary = f.read()

    # Inject the Confident AI Link if it exists
    if results.confident_link:
        markdown_summary += (
            f"\n\n---\n"
            f"### 🔍 Deep Dive\n"
            f"[**View full trace and logs on Confident AI**]({results.confident_link})"
        )

    # 3. Try to post it to GitHub (Snippet provided below)
    _post_github_pr_comment(markdown_summary)


    if not passed:
        print("Test run failed")
        sys.exit(1)
    else:
        print("Test run passed")
        sys.exit(0)


## Here goes all the helper methods for the cicd command
def _invoke_model_callback(model_callback: Any, user_input: str) -> Any:
    if inspect.iscoroutinefunction(model_callback):
        loop = get_or_create_event_loop()
        return loop.run_until_complete(model_callback(user_input))
    return model_callback(user_input)


def _turn_from_model_callback_result(raw: Any) -> Turn:
    if isinstance(raw, Turn):
        return raw
    if isinstance(raw, dict):
        data = dict(raw)
        # Assistant response only; role is required on Turn but ignored here.
        data["role"] = "assistant"
        return Turn.model_validate(data)
    raise TypeError(
        "model_callback must return a Turn instance or a dict compatible with "
        f"Turn; got {type(raw).__name__}"
    )


def _llm_test_case_from_golden_and_turn(golden: Golden, turn: Turn) -> LLMTestCase:
    return LLMTestCase(
        input=golden.input,
        actual_output=turn.content,
        expected_output=golden.expected_output,
        context=golden.context,
        retrieval_context=(
            turn.retrieval_context
            if turn.retrieval_context is not None
            else golden.retrieval_context
        ),
        metadata=(
            turn.metadata
            if turn.metadata is not None
            else golden.additional_metadata
        ),
        tools_called=(
            turn.tools_called
            if turn.tools_called is not None
            else golden.tools_called
        ),
        expected_tools=golden.expected_tools,
        comments=golden.comments,
        name=golden.name,
        custom_column_key_values=golden.custom_column_key_values,
        mcp_tools_called=turn.mcp_tools_called,
        mcp_resources_called=turn.mcp_resources_called,
        mcp_prompts_called=turn.mcp_prompts_called,
        multimodal=golden.multimodal,
    )


def _create_test_cases_from_goldens(
    goldens: List[Golden], model_callback: Any
) -> List[LLMTestCase]:
    """Run ``model_callback`` on each golden's input; expect a ``Turn``-shaped response."""
    test_cases: List[LLMTestCase] = []
    for golden in goldens:
        raw = _invoke_model_callback(model_callback, golden.input)
        turn = _turn_from_model_callback_result(raw)
        test_cases.append(_llm_test_case_from_golden_and_turn(golden, turn))
    return test_cases


def _build_metrics_from_config(
    metrics_cfg: Any,
) -> List[Union[BaseMetric, BaseConversationalMetric, BaseArenaMetric]]:
    """
    Build metric instances from YAML-style config: a list of dicts with
    ``name`` (class name under ``deepeval.metrics``) plus constructor kwargs.
    """
    if metrics_cfg is None:
        return []
    if not isinstance(metrics_cfg, list):
        raise TypeError(
            "metrics must be a list of dicts with a 'name' field and optional kwargs"
        )
    built: List[Union[BaseMetric, BaseConversationalMetric, BaseArenaMetric]] = []
    for i, item in enumerate(metrics_cfg):
        if not isinstance(item, dict):
            raise TypeError(f"metrics[{i}] must be a dict, got {type(item).__name__}")
        class_name = item.get("name")
        if not class_name or not isinstance(class_name, str):
            raise ValueError(
                f"metrics[{i}] requires a non-empty string 'name' (metric class name)"
            )
        cls = getattr(metrics_module, class_name, None)
        if cls is None:
            raise ValueError(
                f"Unknown metric '{class_name}'. "
                "It must be importable from deepeval.metrics."
            )
        kwargs = {k: v for k, v in item.items() if k != "name"}
        try:
            built.append(cls(**kwargs))
        except TypeError as e:
            raise TypeError(
                f"Failed to construct {class_name} with kwargs {kwargs!r}: {e}"
            ) from e
    return built


def _load_model_callback_from_file(
    file_path: str, function_name: str = "model_callback"
) -> Any:
    """Load a model callback function from a Python file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model callback file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(
        "deepeval_cicd_model_callback_module", file_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["deepeval_cicd_model_callback_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}"
        )

    callback = getattr(module, function_name)
    if not callable(callback):
        raise TypeError(f"'{function_name}' in {file_path} is not callable")

    return callback


def _load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _post_github_pr_comment(markdown_content: str):
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY") # format: "owner/repo"
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not token or not repo or not event_path:
        print("Not running in a GitHub PR environment or missing GITHUB_TOKEN. Skipping PR comment.")
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

    base_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # 1. Fetch existing comments
        existing_comments = requests.get(base_url, headers=headers).json()
        
        # 2. Look for our specific bot comment
        comment_id_to_update = None
        for comment in existing_comments:
            if "🚀 DeepEval Evaluation Results" in comment.get("body", ""):
                comment_id_to_update = comment["id"]
                break

        # 3. Update or Create
        if comment_id_to_update:
            patch_url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id_to_update}"
            response = requests.patch(patch_url, json={"body": markdown_content}, headers=headers)
            action = "updated"
        else:
            response = requests.post(base_url, json={"body": markdown_content}, headers=headers)
            action = "posted"

        if response.status_code in [200, 201]:
            print(f"✅ Successfully {action} DeepEval results on GitHub PR.")
        else:
            print(f"⚠️ Failed to {action} PR comment. Status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error while posting to GitHub: {e}")
