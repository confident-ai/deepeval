from asyncio import Task
from typing import Iterator, List, Optional, Union, Literal
from dataclasses import dataclass, field
from opentelemetry.trace import Tracer
from opentelemetry.context import Context, attach, detach
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import json
import csv
import webbrowser
import os
import datetime
import time
import ast
import uuid
from opentelemetry import baggage

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.dataset.utils import (
    convert_test_cases_to_goldens,
    convert_goldens_to_test_cases,
    convert_convo_goldens_to_convo_test_cases,
    convert_convo_test_cases_to_convo_goldens,
    format_turns,
    check_tracer,
    parse_turns,
    trimAndLoadJson,
)
from deepeval.dataset.api import (
    APIDataset,
    DatasetHttpResponse,
    APIQueueDataset,
)
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.metrics.base_metric import BaseMetric
from deepeval.telemetry import capture_evaluation_run, capture_pull_dataset
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
)
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.test_run.test_run import TEMP_FILE_PATH
from deepeval.utils import (
    convert_keys_to_snake_case,
    get_or_create_event_loop,
    open_browser,
)
from deepeval.test_run import (
    global_test_run_manager,
)
from deepeval.dataset.types import global_evaluation_tasks
from deepeval.openai.utils import openai_test_case_pairs
from deepeval.tracing import trace_manager
from deepeval.tracing.tracing import EVAL_DUMMY_SPAN_NAME


valid_file_types = ["csv", "json", "jsonl"]


@dataclass
class EvaluationDataset:
    _multi_turn: bool = field(default=False)
    _alias: Union[str, None] = field(default=None)
    _id: Union[str, None] = field(default=None)

    _goldens: List[Golden] = field(default_factory=[], repr=None)
    _conversational_goldens: List[ConversationalGolden] = field(
        default_factory=[], repr=None
    )

    _llm_test_cases: List[LLMTestCase] = field(default_factory=[], repr=None)
    _conversational_test_cases: List[ConversationalTestCase] = field(
        default_factory=[], repr=None
    )

    def __init__(
        self,
        goldens: Union[List[Golden], List[ConversationalGolden]] = [],
    ):
        self._alias = None
        self._id = None
        if len(goldens) > 0:
            self._multi_turn = (
                True if isinstance(goldens[0], ConversationalGolden) else False
            )

        self._goldens = []
        self._conversational_goldens = []
        for golden in goldens:
            golden._dataset_rank = len(goldens)
            if self._multi_turn:
                self._add_conversational_golden(golden)
            else:
                self._add_golden(golden)

        self._llm_test_cases = []
        self._conversational_test_cases = []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(test_cases={self.test_cases}, "
            f"goldens={self.goldens}, "
            f"_alias={self._alias}, _id={self._id}, _multi_turn={self._multi_turn})"
        )

    @property
    def goldens(self) -> Union[List[Golden], List[ConversationalGolden]]:
        if self._multi_turn:
            return self._conversational_goldens

        return self._goldens

    @goldens.setter
    def goldens(
        self,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ):
        goldens_list = self._goldens
        conversational_goldens_list = self._conversational_goldens
        self._goldens = []
        self._conversational_goldens = []
        try:
            for golden in goldens:
                if not isinstance(golden, Golden) and not isinstance(
                    golden, ConversationalGolden
                ):
                    raise TypeError(
                        "Your goldens must be instances of either ConversationalGolden or Golden"
                    )

                golden._dataset_alias = self._alias
                golden._dataset_id = self._id
                golden._dataset_rank = len(goldens)
                if self._multi_turn:
                    self._add_conversational_golden(golden)
                else:
                    self.add_golden(golden)
        except Exception as e:
            self._goldens = goldens_list
            self._conversational_goldens = conversational_goldens_list
            raise e

    @property
    def test_cases(
        self,
    ) -> Union[List[LLMTestCase], List[ConversationalTestCase]]:
        if self._multi_turn:
            return self._conversational_test_cases

        return self._llm_test_cases

    @test_cases.setter
    def test_cases(
        self,
        test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    ):
        llm_test_cases = []
        conversational_test_cases = []
        for test_case in test_cases:
            if not isinstance(test_case, LLMTestCase) and not isinstance(
                test_case, ConversationalTestCase
            ):
                continue

            test_case._dataset_alias = self._alias
            test_case._dataset_id = self._id
            if isinstance(test_case, LLMTestCase):
                test_case._dataset_rank = len(llm_test_cases)
                llm_test_cases.append(test_case)
            elif isinstance(test_case, ConversationalTestCase):
                test_case._dataset_rank = len(conversational_test_cases)
                conversational_test_cases.append(test_case)

        self._llm_test_cases = llm_test_cases
        self._conversational_test_cases = conversational_test_cases

    def add_test_case(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ):
        test_case._dataset_alias = self._alias
        test_case._dataset_id = self._id
        if isinstance(test_case, LLMTestCase):
            test_case._dataset_rank = len(self._llm_test_cases)
            self._llm_test_cases.append(test_case)
        elif isinstance(test_case, ConversationalTestCase):
            test_case._dataset_rank = len(self._conversational_test_cases)
            self._conversational_test_cases.append(test_case)

    def add_golden(self, golden: Union[Golden, ConversationalGolden]):
        if self._multi_turn:
            self._add_conversational_golden(golden)
        else:
            self._add_golden(golden)

    def _add_golden(self, golden: Union[Golden, ConversationalGolden]):
        if isinstance(golden, Golden):
            self._goldens.append(golden)
        else:
            raise TypeError(
                "You cannot add a multi-turn ConversationalGolden to a single-turn dataset. You can only add a Golden."
            )

    def _add_conversational_golden(
        self, golden: Union[Golden, ConversationalGolden]
    ):
        if isinstance(golden, ConversationalGolden):
            self._conversational_goldens.append(golden)
        else:
            raise TypeError(
                "You cannot add a single-turn Golden to a multi-turn dataset. You can only add a ConversationalGolden."
            )

    def add_test_cases_from_csv_file(
        self,
        file_path: str,
        input_col_name: str,
        actual_output_col_name: str,
        expected_output_col_name: Optional[str] = None,
        context_col_name: Optional[str] = None,
        context_col_delimiter: str = ";",
        retrieval_context_col_name: Optional[str] = None,
        retrieval_context_col_delimiter: str = ";",
        tools_called_col_name: Optional[str] = None,
        tools_called_col_delimiter: str = ";",
        expected_tools_col_name: Optional[str] = None,
        expected_tools_col_delimiter: str = ";",
        additional_metadata_col_name: Optional[str] = None,
    ):
        """
        Load test cases from a CSV file.

        This method reads a CSV file, extracting test case data based on specified column names. It creates LLMTestCase objects for each row in the CSV and adds them to the Dataset instance. The context data, if provided, is expected to be a delimited string in the CSV, which this method will parse into a list.

        Args:
            file_path (str): Path to the CSV file containing the test cases.
            input_col_name (str): The column name in the CSV corresponding to the input for the test case.
            actual_output_col_name (str): The column name in the CSV corresponding to the actual output for the test case.
            expected_output_col_name (str, optional): The column name in the CSV corresponding to the expected output for the test case. Defaults to None.
            context_col_name (str, optional): The column name in the CSV corresponding to the context for the test case. Defaults to None.
            context_delimiter (str, optional): The delimiter used to separate items in the context list within the CSV file. Defaults to ';'.
            retrieval_context_col_name (str, optional): The column name in the CSV corresponding to the retrieval context for the test case. Defaults to None.
            retrieval_context_delimiter (str, optional): The delimiter used to separate items in the retrieval context list within the CSV file. Defaults to ';'.
            additional_metadata_col_name (str, optional): The column name in the CSV corresponding to additional metadata for the test case. Defaults to None.

        Returns:
            None: The method adds test cases to the Dataset instance but does not return anything.

        Raises:
            FileNotFoundError: If the CSV file specified by `file_path` cannot be found.
            pd.errors.EmptyDataError: If the CSV file is empty.
            KeyError: If one or more specified columns are not found in the CSV file.

        Note:
            The CSV file is expected to contain columns as specified in the arguments. Each row in the file represents a single test case. The method assumes the file is properly formatted and the specified columns exist. For context data represented as lists in the CSV, ensure the correct delimiter is specified.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )

        def get_column_data(df: pd.DataFrame, col_name: str, default=None):
            return (
                df[col_name].values
                if col_name in df.columns
                else [default] * len(df)
            )

        df = pd.read_csv(file_path)
        # Convert np.nan (default for missing values in pandas) to None for compatibility with Python and Pydantic
        df = df.astype(object).where(pd.notna(df), None)

        inputs = get_column_data(df, input_col_name)
        actual_outputs = get_column_data(df, actual_output_col_name)
        expected_outputs = get_column_data(
            df, expected_output_col_name, default=None
        )
        contexts = [
            context.split(context_col_delimiter) if context else []
            for context in get_column_data(df, context_col_name, default="")
        ]
        retrieval_contexts = [
            (
                retrieval_context.split(retrieval_context_col_delimiter)
                if retrieval_context
                else []
            )
            for retrieval_context in get_column_data(
                df, retrieval_context_col_name, default=""
            )
        ]
        tools_called = []
        for tools_called_json in get_column_data(
            df, tools_called_col_name, default="[]"
        ):
            if tools_called_json:
                try:
                    parsed_tools = [
                        ToolCall(**tool)
                        for tool in trimAndLoadJson(tools_called_json)
                    ]
                    tools_called.append(parsed_tools)
                except ValueError as e:
                    raise ValueError(f"Error processing tools_called: {e}")
            else:
                tools_called.append([])

        expected_tools = []
        for expected_tools_json in get_column_data(
            df, expected_tools_col_name, default="[]"
        ):
            if expected_tools_json:
                try:
                    parsed_tools = [
                        ToolCall(**tool)
                        for tool in trimAndLoadJson(expected_tools_json)
                    ]
                    expected_tools.append(parsed_tools)
                except ValueError as e:
                    raise ValueError(f"Error processing expected_tools: {e}")
            else:
                expected_tools.append([])
        additional_metadatas = [
            ast.literal_eval(metadata) if metadata else None
            for metadata in get_column_data(
                df, additional_metadata_col_name, default=""
            )
        ]

        for (
            input,
            actual_output,
            expected_output,
            context,
            retrieval_context,
            tools_called,
            expected_tools,
            additional_metadata,
        ) in zip(
            inputs,
            actual_outputs,
            expected_outputs,
            contexts,
            retrieval_contexts,
            tools_called,
            expected_tools,
            additional_metadatas,
        ):
            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                    additional_metadata=additional_metadata,
                )
            )

    def add_test_cases_from_json_file(
        self,
        file_path: str,
        input_key_name: str,
        actual_output_key_name: str,
        expected_output_key_name: Optional[str] = None,
        context_key_name: Optional[str] = None,
        retrieval_context_key_name: Optional[str] = None,
        tools_called_key_name: Optional[str] = None,
        expected_tools_key_name: Optional[str] = None,
        encoding_type: str = "utf-8",
    ):
        """
        Load test cases from a JSON file.

        This method reads a JSON file containing a list of objects, each representing a test case. It extracts the necessary information based on specified key names and creates LLMTestCase objects to add to the Dataset instance.

        Args:
            file_path (str): Path to the JSON file containing the test cases.
            input_key_name (str): The key name in the JSON objects corresponding to the input for the test case.
            actual_output_key_name (str): The key name in the JSON objects corresponding to the actual output for the test case.
            expected_output_key_name (str, optional): The key name in the JSON objects corresponding to the expected output for the test case. Defaults to None.
            context_key_name (str, optional): The key name in the JSON objects corresponding to the context for the test case. Defaults to None.
            retrieval_context_key_name (str, optional): The key name in the JSON objects corresponding to the retrieval context for the test case. Defaults to None.

        Returns:
            None: The method adds test cases to the Dataset instance but does not return anything.

        Raises:
            FileNotFoundError: If the JSON file specified by `file_path` cannot be found.
            ValueError: If the JSON file is not valid or if required keys (input and actual output) are missing in one or more JSON objects.

        Note:
            The JSON file should be structured as a list of objects, with each object containing the required keys. The method assumes the file format and keys are correctly defined and present.
        """
        try:
            with open(file_path, "r", encoding=encoding_type) as file:
                json_list = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON file.")

        # Process each JSON object
        for json_obj in json_list:
            if (
                input_key_name not in json_obj
                or actual_output_key_name not in json_obj
            ):
                raise ValueError(
                    "Required fields are missing in one or more JSON objects"
                )

            input = json_obj[input_key_name]
            actual_output = json_obj[actual_output_key_name]
            expected_output = json_obj.get(expected_output_key_name)
            context = json_obj.get(context_key_name)
            retrieval_context = json_obj.get(retrieval_context_key_name)
            tools_called_data = json_obj.get(tools_called_key_name, [])
            tools_called = [ToolCall(**tool) for tool in tools_called_data]
            expected_tools_data = json_obj.get(expected_tools_key_name, [])
            expected_tools = [ToolCall(**tool) for tool in expected_tools_data]

            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                )
            )

    def add_goldens_from_csv_file(
        self,
        file_path: str,
        input_col_name: Optional[str] = "input",
        actual_output_col_name: Optional[str] = "actual_output",
        expected_output_col_name: Optional[str] = "expected_output",
        context_col_name: Optional[str] = "context",
        context_col_delimiter: str = "|",
        retrieval_context_col_name: Optional[str] = "retrieval_context",
        retrieval_context_col_delimiter: str = "|",
        tools_called_col_name: Optional[str] = "tools_called",
        tools_called_col_delimiter: str = ";",
        expected_tools_col_name: Optional[str] = "expected_tools",
        expected_tools_col_delimiter: str = ";",
        source_file_col_name: Optional[str] = None,
        additional_metadata_col_name: Optional[str] = None,
        scenario_col_name: Optional[str] = "scenario",
        turns_col_name: Optional[str] = "turns",
        expected_outcome_col_name: Optional[str] = "expected_outcome",
        user_description_col_name: Optional[str] = "user_description",
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )

        def get_column_data(df: pd.DataFrame, col_name: str, default=None):
            return (
                df[col_name].values
                if col_name in df.columns
                else [default] * len(df)
            )

        df = (
            pd.read_csv(file_path)
            .astype(object)
            .where(pd.notna(pd.read_csv(file_path)), None)
        )

        inputs = get_column_data(df, input_col_name)
        actual_outputs = get_column_data(
            df, actual_output_col_name, default=None
        )
        expected_outputs = get_column_data(
            df, expected_output_col_name, default=None
        )
        contexts = [
            context.split(context_col_delimiter) if context else []
            for context in get_column_data(df, context_col_name, default="")
        ]
        retrieval_contexts = [
            (
                retrieval_context.split(retrieval_context_col_delimiter)
                if retrieval_context
                else []
            )
            for retrieval_context in get_column_data(
                df, retrieval_context_col_name, default=""
            )
        ]
        tools_called = [
            (
                tool_called.split(tools_called_col_delimiter)
                if tool_called
                else []
            )
            for tool_called in get_column_data(
                df, tools_called_col_name, default=""
            )
        ]
        expected_tools = [
            (
                expected_tool.split(expected_tools_col_delimiter)
                if expected_tool
                else []
            )
            for expected_tool in get_column_data(
                df, expected_tools_col_name, default=""
            )
        ]
        source_files = get_column_data(df, source_file_col_name)
        additional_metadatas = [
            ast.literal_eval(metadata) if metadata else None
            for metadata in get_column_data(
                df, additional_metadata_col_name, default=""
            )
        ]
        scenarios = get_column_data(df, scenario_col_name)
        turns_raw = get_column_data(df, turns_col_name)
        expected_outcomes = get_column_data(df, expected_outcome_col_name)
        user_descriptions = get_column_data(df, user_description_col_name)

        for (
            input,
            actual_output,
            expected_output,
            context,
            retrieval_context,
            tools_called,
            expected_tools,
            source_file,
            additional_metadata,
            scenario,
            turns,
            expected_outcome,
            user_description,
        ) in zip(
            inputs,
            actual_outputs,
            expected_outputs,
            contexts,
            retrieval_contexts,
            tools_called,
            expected_tools,
            source_files,
            additional_metadatas,
            scenarios,
            turns_raw,
            expected_outcomes,
            user_descriptions,
        ):
            if scenario:
                self._multi_turn = True
                parsed_turns = parse_turns(turns)
                self.goldens.append(
                    ConversationalGolden(
                        scenario=scenario,
                        turns=parsed_turns,
                        expected_outcome=expected_outcome,
                        user_description=user_description,
                        context=context,
                    )
                )
            else:
                self._multi_turn = False
                self.goldens.append(
                    Golden(
                        input=input,
                        actual_output=actual_output,
                        expected_output=expected_output,
                        context=context,
                        retrieval_context=retrieval_context,
                        tools_called=tools_called,
                        expected_tools=expected_tools,
                        additional_metadata=additional_metadata,
                        source_file=source_file,
                    )
                )

    def add_goldens_from_json_file(
        self,
        file_path: str,
        input_key_name: str = "input",
        actual_output_key_name: Optional[str] = "actual_output",
        expected_output_key_name: Optional[str] = "expected_output",
        context_key_name: Optional[str] = "context",
        retrieval_context_key_name: Optional[str] = "retrieval_context",
        tools_called_key_name: Optional[str] = "tools_called",
        expected_tools_key_name: Optional[str] = "expected_tools",
        source_file_key_name: Optional[str] = "source_file",
        additional_metadata_key_name: Optional[str] = "additional_metadata",
        scenario_key_name: Optional[str] = "scenario",
        turns_key_name: Optional[str] = "turns",
        expected_outcome_key_name: Optional[str] = "expected_outcome",
        user_description_key_name: Optional[str] = "user_description",
        encoding_type: str = "utf-8",
    ):
        try:
            with open(file_path, "r", encoding=encoding_type) as file:
                json_list = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON file.")

        for json_obj in json_list:
            if scenario_key_name in json_obj and json_obj[scenario_key_name]:
                scenario = json_obj.get(scenario_key_name)
                turns = json_obj.get(turns_key_name, [])
                expected_outcome = json_obj.get(expected_outcome_key_name)
                user_description = json_obj.get(user_description_key_name)
                context = json_obj.get(context_key_name)

                parsed_turns = parse_turns(turns) if turns else []

                self._multi_turn = True
                self.goldens.append(
                    ConversationalGolden(
                        scenario=scenario,
                        turns=parsed_turns,
                        expected_outcome=expected_outcome,
                        user_description=user_description,
                        context=context,
                    )
                )
            else:
                input = json_obj.get(input_key_name)
                actual_output = json_obj.get(actual_output_key_name)
                expected_output = json_obj.get(expected_output_key_name)
                context = json_obj.get(context_key_name)
                retrieval_context = json_obj.get(retrieval_context_key_name)
                tools_called = json_obj.get(tools_called_key_name)
                expected_tools = json_obj.get(expected_tools_key_name)
                source_file = json_obj.get(source_file_key_name)
                additional_metadata = json_obj.get(additional_metadata_key_name)

                self._multi_turn = False
                self.goldens.append(
                    Golden(
                        input=input,
                        actual_output=actual_output,
                        expected_output=expected_output,
                        context=context,
                        retrieval_context=retrieval_context,
                        tools_called=tools_called,
                        expected_tools=expected_tools,
                        additional_metadata=additional_metadata,
                        source_file=source_file,
                    )
                )

    def push(
        self,
        alias: str,
        finalized: bool = True,
    ):
        if len(self.goldens) == 0:
            raise ValueError(
                "Unable to push empty dataset to Confident AI, there must be at least one golden in dataset."
            )

        api = Api()
        api_dataset = APIDataset(
            goldens=self.goldens if not self._multi_turn else None,
            conversationalGoldens=(self.goldens if self._multi_turn else None),
            finalized=finalized,
        )
        try:
            body = api_dataset.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_dataset.dict(by_alias=True, exclude_none=True)

        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.DATASET_ALIAS_ENDPOINT,
            body=body,
            url_params={"alias": alias},
        )
        if link:
            console = Console()
            console.print(
                "✅ Dataset successfully pushed to Confident AI! View at "
                f"[link={link}]{link}[/link]"
            )
            open_browser(link)

    def pull(
        self,
        alias: str,
        finalized: bool = True,
        auto_convert_goldens_to_test_cases: bool = False,
        public: bool = False,
    ):
        api = Api()
        with capture_pull_dataset():
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Pulling [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] from Confident AI...",
                    total=100,
                )
                start_time = time.perf_counter()
                data, _ = api.send_request(
                    method=HttpMethods.GET,
                    endpoint=Endpoints.DATASET_ALIAS_ENDPOINT,
                    url_params={"alias": alias},
                    params={
                        "finalized": str(finalized).lower(),
                        "public": str(public).lower(),
                    },
                )

                response = DatasetHttpResponse(
                    id=data["id"],
                    goldens=convert_keys_to_snake_case(
                        data.get("goldens", None)
                    ),
                    conversationalGoldens=convert_keys_to_snake_case(
                        data.get("conversationalGoldens", None)
                    ),
                )

                self._alias = alias
                self._id = response.id
                self._multi_turn = response.goldens is None
                self.goldens = []
                self.test_cases = []

                if auto_convert_goldens_to_test_cases:
                    if not self._multi_turn:
                        llm_test_cases = convert_goldens_to_test_cases(
                            response.goldens, alias, response.id
                        )
                        self._llm_test_cases.extend(llm_test_cases)
                    else:
                        conversational_test_cases = (
                            convert_convo_goldens_to_convo_test_cases(
                                response.conversational_goldens,
                                alias,
                                response.id,
                            )
                        )
                        self._conversational_test_cases.extend(
                            conversational_test_cases
                        )
                else:
                    if not self._multi_turn:
                        self.goldens = response.goldens
                    else:
                        self.goldens = response.conversational_goldens

                    for golden in self.goldens:
                        golden._dataset_alias = alias
                        golden._dataset_id = response.id

                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)",
                    completed=100,
                )

    def queue(
        self,
        alias: str,
        goldens: Union[List[Golden], List[ConversationalGolden]],
        print_response: bool = True,
    ):
        if len(goldens) == 0:
            raise ValueError(
                f"Can't queue empty list of goldens to dataset with alias: {alias} on Confident AI."
            )
        api = Api()

        multi_turn = isinstance(goldens[0], ConversationalGolden)

        api_dataset = APIQueueDataset(
            alias=alias,
            goldens=goldens if not multi_turn else None,
            conversationalGoldens=goldens if multi_turn else None,
        )
        try:
            body = api_dataset.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_dataset.dict(by_alias=True, exclude_none=True)

        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.DATASET_ALIAS_QUEUE_ENDPOINT,
            body=body,
            url_params={"alias": alias},
        )
        if link and print_response:
            console = Console()
            console.print(
                "✅ Goldens successfully queued to Confident AI! Annotate & finalized them at "
                f"[link={link}]{link}[/link]"
            )

    def delete(
        self,
        alias: str,
    ):
        api = Api()
        api.send_request(
            method=HttpMethods.DELETE,
            endpoint=Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": alias},
        )
        console = Console()
        console.print("✅ Dataset successfully deleted from Confident AI!")

    def generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        context_construction_config=None,
        synthesizer=None,
    ):
        from deepeval.synthesizer import Synthesizer
        from deepeval.synthesizer.config import ContextConstructionConfig

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        if context_construction_config is not None:
            assert isinstance(
                context_construction_config, ContextConstructionConfig
            )

        self.goldens.extend(
            synthesizer.generate_goldens_from_docs(
                document_paths=document_paths,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                context_construction_config=context_construction_config,
                _send_data=False,
            )
        )

    def generate_goldens_from_contexts(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = True,
        max_goldens_per_context: int = 2,
        synthesizer=None,
    ):
        from deepeval.synthesizer import Synthesizer

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        self.goldens.extend(
            synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                _send_data=False,
            )
        )

    def generate_goldens_from_scratch(
        self,
        num_goldens: int,
        synthesizer=None,
    ):
        from deepeval.synthesizer import Synthesizer

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        self.goldens.extend(
            synthesizer.generate_goldens_from_scratch(
                num_goldens=num_goldens,
                _send_data=False,
            )
        )

    def save_as(
        self,
        file_type: Literal["json", "csv", "jsonl"],
        directory: str,
        file_name: Optional[str] = None,
        include_test_cases: bool = False,
    ) -> str:
        if file_type not in valid_file_types:
            raise ValueError(
                f"Invalid file type. Available file types to save as: {', '.join(type for type in valid_file_types)}"
            )

        if self._multi_turn:
            goldens = [
                ConversationalGolden(
                    scenario=golden.scenario,
                    turns=golden.turns,
                    expected_outcome=golden.expected_outcome,
                    user_description=golden.user_description,
                    context=golden.context,
                )
                for golden in self.goldens
            ]
        else:
            goldens = [
                Golden(
                    input=golden.input,
                    expected_output=golden.expected_output,
                    actual_output=golden.actual_output,
                    retrieval_context=golden.retrieval_context,
                    context=golden.context,
                    source_file=golden.source_file,
                )
                for golden in self.goldens
            ]
        if include_test_cases:
            if self._multi_turn:
                goldens.extend(
                    convert_convo_test_cases_to_convo_goldens(self.test_cases)
                )
            else:
                goldens.extend(convert_test_cases_to_goldens(self.test_cases))

        if len(goldens) == 0:
            raise ValueError(
                f"No goldens found. Please generate goldens before attempting to save data as {file_type}"
            )

        new_filename = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if file_name is None
            else file_name
        ) + f".{file_type}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        full_file_path = os.path.join(directory, new_filename)

        if file_type == "json":
            with open(full_file_path, "w", encoding="utf-8") as file:
                if self._multi_turn:
                    json_data = [
                        {
                            "scenario": golden.scenario,
                            "turns": (
                                format_turns(golden.turns)
                                if golden.turns
                                else None
                            ),
                            "expected_outcome": golden.expected_outcome,
                            "user_description": golden.user_description,
                            "context": golden.context,
                        }
                        for golden in goldens
                    ]
                else:
                    json_data = [
                        {
                            "input": golden.input,
                            "actual_output": golden.actual_output,
                            "expected_output": golden.expected_output,
                            "retrieval_context": golden.retrieval_context,
                            "context": golden.context,
                            "source_file": golden.source_file,
                        }
                        for golden in goldens
                    ]
                json.dump(json_data, file, indent=4, ensure_ascii=False)
        elif file_type == "csv":
            with open(
                full_file_path, "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                if self._multi_turn:
                    writer.writerow(
                        [
                            "scenario",
                            "turns",
                            "expected_outcome",
                            "user_description",
                            "context",
                        ]
                    )
                    for golden in goldens:
                        context = (
                            "|".join(golden.context)
                            if golden.context is not None
                            else None
                        )
                        turns = (
                            format_turns(golden.turns)
                            if golden.turns is not None
                            else None
                        )
                        writer.writerow(
                            [
                                golden.scenario,
                                turns,
                                golden.expected_outcome,
                                golden.user_description,
                                context,
                            ]
                        )
                else:
                    writer.writerow(
                        [
                            "input",
                            "actual_output",
                            "expected_output",
                            "retrieval_context",
                            "context",
                            "source_file",
                        ]
                    )
                    for golden in goldens:
                        retrieval_context = (
                            "|".join(golden.retrieval_context)
                            if golden.retrieval_context is not None
                            else None
                        )
                        context = (
                            "|".join(golden.context)
                            if golden.context is not None
                            else None
                        )
                        writer.writerow(
                            [
                                golden.input,
                                golden.actual_output,
                                golden.expected_output,
                                retrieval_context,
                                context,
                                golden.source_file,
                            ]
                        )
        elif file_type == "jsonl":
            with open(full_file_path, "w", encoding="utf-8") as file:
                for golden in goldens:
                    if self._multi_turn:
                        turns = (
                            format_turns(golden.turns) if golden.turns else None
                        )
                        record = {
                            "scenario": golden.scenario,
                            "turns": turns,
                            "expected_outcome": golden.expected_outcome,
                            "user_description": golden.user_description,
                            "context": golden.context,
                        }
                    else:
                        retrieval_context = (
                            "|".join(golden.retrieval_context)
                            if golden.retrieval_context is not None
                            else None
                        )
                        context = (
                            "|".join(golden.context)
                            if golden.context is not None
                            else None
                        )
                        record = {
                            "input": golden.input,
                            "actual_output": golden.actual_output,
                            "expected_output": golden.expected_output,
                            "retrieval_context": retrieval_context,
                            "context": context,
                        }

                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Evaluation dataset saved at {full_file_path}!")
        return full_file_path

    def evals_iterator(
        self,
        metrics: Optional[List[BaseMetric]] = None,
        identifier: Optional[str] = None,
        display_config: Optional["DisplayConfig"] = None,
        cache_config: Optional["CacheConfig"] = None,
        error_config: Optional["ErrorConfig"] = None,
        async_config: Optional["AsyncConfig"] = None,
        run_otel: Optional[bool] = False,
    ) -> Iterator[Golden]:
        from deepeval.evaluate.utils import (
            aggregate_metric_pass_rates,
            print_test_result,
            write_test_result_to_file,
        )
        from deepeval.evaluate.types import EvaluationResult, TestResult
        from deepeval.evaluate.execute import (
            a_execute_agentic_test_cases_from_loop,
            execute_agentic_test_cases_from_loop,
        )
        from deepeval.evaluate.configs import (
            AsyncConfig,
            DisplayConfig,
            CacheConfig,
            ErrorConfig,
        )

        if display_config is None:
            display_config: DisplayConfig = DisplayConfig()
        if cache_config is None:
            cache_config: CacheConfig = CacheConfig()
        if error_config is None:
            error_config: ErrorConfig = ErrorConfig()
        if async_config is None:
            async_config: AsyncConfig = AsyncConfig()

        if not self.goldens or len(self.goldens) == 0:
            raise ValueError("Unable to evaluate dataset with no goldens.")
        trace_manager.integration_traces_to_evaluate.clear()
        goldens = self.goldens
        with capture_evaluation_run("traceable evaluate()"):
            global_test_run_manager.reset()
            start_time = time.perf_counter()
            test_results: List[TestResult] = []

            # sandwich start trace for OTEL
            if run_otel:
                ctx = self._start_otel_test_run()  # ignored span
                ctx_token = attach(ctx)

            if async_config.run_async:
                loop = get_or_create_event_loop()
                for golden in a_execute_agentic_test_cases_from_loop(
                    goldens=goldens,
                    identifier=identifier,
                    loop=loop,
                    trace_metrics=metrics,
                    test_results=test_results,
                    display_config=display_config,
                    cache_config=cache_config,
                    error_config=error_config,
                    async_config=async_config,
                ):
                    if run_otel:
                        _tracer = check_tracer()
                        with _tracer.start_as_current_span(
                            name=EVAL_DUMMY_SPAN_NAME,
                            context=ctx,
                        ):
                            yield golden
                    else:
                        yield golden

            else:
                for golden in execute_agentic_test_cases_from_loop(
                    goldens=goldens,
                    trace_metrics=metrics,
                    display_config=display_config,
                    cache_config=cache_config,
                    error_config=error_config,
                    test_results=test_results,
                    identifier=identifier,
                ):
                    if run_otel:
                        _tracer = check_tracer()
                        with _tracer.start_as_current_span(
                            name=EVAL_DUMMY_SPAN_NAME,
                            context=ctx,
                        ):
                            yield golden
                    else:
                        yield golden

            end_time = time.perf_counter()
            run_duration = end_time - start_time
            if display_config.print_results:
                for test_result in test_results:
                    print_test_result(
                        test_result, display_config.display_option
                    )
                aggregate_metric_pass_rates(test_results)
            if display_config.file_output_dir is not None:
                for test_result in test_results:
                    write_test_result_to_file(
                        test_result,
                        display_config.display_option,
                        display_config.file_output_dir,
                    )

            # update hyperparameters
            test_run = global_test_run_manager.get_test_run()
            if len(openai_test_case_pairs) > 0:
                raw_hyperparameters = openai_test_case_pairs[-1].hyperparameters
                test_run.hyperparameters = process_hyperparameters(
                    raw_hyperparameters
                )

            # clean up
            openai_test_case_pairs.clear()
            global_test_run_manager.save_test_run(TEMP_FILE_PATH)

            # sandwich end trace for OTEL
            if run_otel:
                self._end_otel_test_run(ctx)
                detach(ctx_token)

            else:
                confident_link = global_test_run_manager.wrap_up_test_run(
                    run_duration, display_table=False
                )
                return EvaluationResult(
                    test_results=test_results, confident_link=confident_link
                )

    def evaluate(self, task: Task):
        global_evaluation_tasks.append(task)

    def _start_otel_test_run(self, tracer: Optional[Tracer] = None) -> Context:
        _tracer = check_tracer(tracer)
        run_id = str(uuid.uuid4())
        print("Starting OTLP test run with run_id: ", run_id)
        ctx = baggage.set_baggage(
            "confident.test_run.id", run_id, context=Context()
        )
        with _tracer.start_as_current_span(
            "start_otel_test_run", context=ctx
        ) as span:
            span.set_attribute("confident.test_run.id", run_id)
        return ctx

    def _end_otel_test_run(self, ctx: Context, tracer: Optional[Tracer] = None):
        run_id = baggage.get_baggage("confident.test_run.id", context=ctx)
        print("Ending OTLP test run with run_id: ", run_id)
        _tracer = check_tracer(tracer)
        with _tracer.start_as_current_span(
            "stop_otel_test_run", context=ctx
        ) as span:
            span.set_attribute("confident.test_run.id", run_id)
