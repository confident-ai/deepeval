from enum import Enum
from typing import List, Optional, Union, Literal
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import json
import csv
import webbrowser
import os
import datetime
import time
import ast

from deepeval.metrics import (
    BaseConversationalMetric,
    BaseMetric,
)
from deepeval.evaluate.types import EvaluationResult
from deepeval.confident.api import Api, Endpoints, HttpMethods, is_confident
from deepeval.dataset.utils import (
    convert_test_cases_to_goldens,
    convert_goldens_to_test_cases,
    convert_convo_goldens_to_convo_test_cases,
    trimAndLoadJson,
)
from deepeval.dataset.api import (
    APIDataset,
    CreateDatasetHttpResponse,
    DatasetHttpResponse,
    APIQueueDataset,
)
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.telemetry import capture_pull_dataset
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
)
from deepeval.utils import convert_keys_to_snake_case

valid_file_types = ["csv", "json", "jsonl"]


@dataclass
class EvaluationDataset:
    _multi_turn: bool = field(default=False)
    _alias: Union[str, None] = field(default=None)
    _id: Union[str, None] = field(default=None)
    _confident_api_key: Optional[str] = None

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

    def __len__(self):
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)

    def evaluate(
        self,
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
    ) -> EvaluationResult:
        from deepeval import evaluate

        if len(self.test_cases) == 0:
            raise ValueError(
                "No test cases found in evaluation dataset. Unable to evaluate empty dataset."
            )

        return evaluate(self.test_cases, metrics)

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
        input_col_name: str,
        actual_output_col_name: Optional[str] = None,
        expected_output_col_name: Optional[str] = None,
        context_col_name: Optional[str] = None,
        context_col_delimiter: str = ";",
        retrieval_context_col_name: Optional[str] = None,
        retrieval_context_col_delimiter: str = ";",
        tools_called_col_name: Optional[str] = None,
        tools_called_col_delimiter: str = ";",
        expected_tools_col_name: Optional[str] = None,
        expected_tools_col_delimiter: str = ";",
        source_file_col_name: Optional[str] = None,
        additional_metadata_col_name: Optional[str] = None,
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
        ):
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
        input_key_name: str,
        actual_output_key_name: Optional[str] = None,
        expected_output_key_name: Optional[str] = None,
        context_key_name: Optional[str] = None,
        retrieval_context_key_name: Optional[str] = None,
        tools_called_key_name: Optional[str] = None,
        expected_tools_key_name: Optional[str] = None,
        source_file_key_name: Optional[str] = None,
        encoding_type: str = "utf-8",
    ):
        try:
            with open(file_path, "r", encoding=encoding_type) as file:
                json_list = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON file.")

        # Process each JSON object
        for json_obj in json_list:
            if input_key_name not in json_obj:
                raise ValueError(
                    "Required fields are missing in one or more JSON objects"
                )

            input = json_obj[input_key_name]
            actual_output = json_obj.get(actual_output_key_name)
            expected_output = json_obj.get(expected_output_key_name)
            context = json_obj.get(context_key_name)
            retrieval_context = json_obj.get(retrieval_context_key_name)
            tools_called = json_obj.get(tools_called_key_name)
            expected_tools = json_obj.get(expected_tools_key_name)
            source_file = json_obj.get(source_file_key_name)

            self.goldens.append(
                Golden(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                    source_file=source_file,
                )
            )

    def push(
        self,
        alias: str,
        overwrite: bool = False,
    ):
        if len(self.goldens) == 0:
            raise ValueError(
                "Unable to push empty dataset to Confident AI, there must be at least one golden in dataset."
            )

        if is_confident():
            api_dataset = APIDataset(
                alias=alias,
                overwrite=overwrite,
                goldens=self.goldens if not self._multi_turn else None,
                conversationalGoldens=(
                    self.goldens if self._multi_turn else None
                ),
            )
            try:
                body = api_dataset.model_dump(by_alias=True, exclude_none=True)
            except AttributeError:
                # Pydantic version below 2.0
                body = api_dataset.dict(by_alias=True, exclude_none=True)

            api = Api()
            result = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.DATASET_ENDPOINT,
                body=body,
            )
            if result:
                response = CreateDatasetHttpResponse(
                    link=result["link"],
                )
                link = response.link
                console = Console()
                console.print(
                    "✅ Dataset successfully pushed to Confident AI! View at "
                    f"[link={link}]{link}[/link]"
                )
                webbrowser.open(link)
        else:
            raise Exception(
                "To push dataset to Confident AI, run `deepeval login`"
            )

    def pull(
        self,
        alias: str,
        finalized: bool = True,
        auto_convert_goldens_to_test_cases: bool = False,
    ):
        with capture_pull_dataset():
            if is_confident() or self._confident_api_key is not None:
                api = Api(api_key=self._confident_api_key)
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
                    result = api.send_request(
                        method=HttpMethods.GET,
                        endpoint=Endpoints.DATASET_ENDPOINT,
                        params={
                            "alias": alias,
                            "finalized": str(finalized).lower(),
                        },
                    )

                    response = DatasetHttpResponse(
                        goldens=convert_keys_to_snake_case(
                            result.get("goldens", None)
                        ),
                        conversationalGoldens=convert_keys_to_snake_case(
                            result.get("conversationalGoldens", None)
                        ),
                        datasetId=result["datasetId"],
                    )

                    self._alias = alias
                    self._id = response.datasetId
                    self._multi_turn = response.goldens is None
                    self.goldens = []
                    self.test_cases = []

                    if auto_convert_goldens_to_test_cases:
                        if not self._multi_turn:
                            llm_test_cases = convert_goldens_to_test_cases(
                                response.goldens, alias, response.datasetId
                            )
                            self._llm_test_cases.extend(llm_test_cases)
                        else:
                            conversational_test_cases = (
                                convert_convo_goldens_to_convo_test_cases(
                                    response.conversational_goldens,
                                    alias,
                                    response.datasetId,
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
                            golden._dataset_id = response.datasetId

                    end_time = time.perf_counter()
                    time_taken = format(end_time - start_time, ".2f")
                    progress.update(
                        task_id,
                        description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)",
                        completed=100,
                    )
            else:
                raise Exception(
                    "Run `deepeval login` to pull dataset from Confident AI"
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

        multi_turn = isinstance(goldens[0], ConversationalGolden)

        if is_confident():
            api = Api()
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

            api = Api()
            result = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.DATASET_QUEUE_ENDPOINT,
                body=body,
                url_params={"alias": alias},
            )
            if result and print_response:
                response = CreateDatasetHttpResponse(
                    link=result["link"],
                )
                link = response.link
                console = Console()
                console.print(
                    "✅ Goldens successfully queued to Confident AI! Annotate & finalized them at "
                    f"[link={link}]{link}[/link]"
                )
        else:
            raise Exception(
                "To push dataset to Confident AI, run `deepeval login`"
            )

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
                    record = {
                        "input": golden.input,
                        "actual_output": golden.actual_output,
                        "expected_output": golden.expected_output,
                        "retrieval_context": golden.retrieval_context,
                        "context": golden.context,
                        "source_file": golden.source_file,
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Evaluation dataset saved at {full_file_path}!")
        return full_file_path
