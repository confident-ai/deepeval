from typing import List, Optional, Union
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import csv
import webbrowser
import os
import datetime
import time
import ast

from deepeval.metrics import BaseMetric
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.dataset.utils import (
    convert_test_cases_to_goldens,
    convert_goldens_to_test_cases,
    convert_convo_goldens_to_convo_test_cases,
)
from deepeval.dataset.api import (
    APIDataset,
    CreateDatasetHttpResponse,
    DatasetHttpResponse,
)
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase
from deepeval.utils import convert_keys_to_snake_case, is_confident
from deepeval.synthesizer.types import *

valid_file_types = ["csv", "json"]


def validate_test_case_type(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    subject: str,
):
    if (
        not isinstance(test_case, LLMTestCase)
        and not isinstance(test_case, ConversationalTestCase)
        and not isinstance(test_case, MLLMTestCase)
    ):
        raise TypeError(
            f"Provided `{subject}` must be a list of LLMTestCase, ConversationalTestCase, or MLLMTestCase"
        )


@dataclass
class EvaluationDataset:
    goldens: List[Golden]
    conversational_goldens: List[ConversationalGolden]
    _alias: Union[str, None] = field(default=None)
    _id: Union[str, None] = field(default=None)
    _llm_test_cases: List[LLMTestCase] = field(default_factory=[], repr=None)
    _conversational_test_cases: List[ConversationalTestCase] = field(
        default_factory=[], repr=None
    )

    def __init__(
        self,
        test_cases: List[
            Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]
        ] = [],
        goldens: List[Golden] = [],
        conversational_goldens: List[ConversationalGolden] = [],
    ):
        for test_case in test_cases:
            validate_test_case_type(test_case, subject="test cases")
        self.goldens = goldens
        self.conversational_goldens = conversational_goldens
        self._alias = None
        self._id = None

        llm_test_cases = []
        conversational_test_cases = []
        mllm_test_cases = []
        for test_case in test_cases:
            if isinstance(test_case, LLMTestCase):
                test_case._dataset_rank = len(llm_test_cases)
                llm_test_cases.append(test_case)
            elif isinstance(test_case, ConversationalTestCase):
                test_case._dataset_rank = len(conversational_test_cases)
                conversational_test_cases.append(test_case)
            elif isinstance(test_case, MLLMTestCase):
                test_case._dataset_rank = len(mllm_test_cases)
                mllm_test_cases.append(test_case)

        self._llm_test_cases = llm_test_cases
        self._conversational_test_cases = conversational_test_cases
        self._mllm_test_cases = mllm_test_cases

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(test_cases={self.test_cases}, "
            f"goldens={self.goldens}, "
            f"conversational_goldens={self.conversational_goldens}, "
            f"_alias={self._alias}, _id={self._id})"
        )

    @property
    def test_cases(
        self,
    ) -> List[Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]]:
        return (
            self._llm_test_cases
            + self._conversational_test_cases
            + self._mllm_test_cases
        )

    @test_cases.setter
    def test_cases(
        self,
        test_cases: List[
            Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]
        ],
    ):
        if not isinstance(test_cases, list):
            raise TypeError("'test_cases' must be a list")

        llm_test_cases = []
        conversational_test_cases = []
        mllm_test_cases = []
        for test_case in test_cases:
            if (
                not isinstance(test_case, LLMTestCase)
                and not isinstance(test_case, ConversationalTestCase)
                and not isinstance(test_case, MLLMTestCase)
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
            elif isinstance(test_case, MLLMTestCase):
                test_case._dataset_rank = len(mllm_test_cases)
                mllm_test_cases.append(test_case)

        self._llm_test_cases = llm_test_cases
        self._conversational_test_cases = conversational_test_cases
        self._mllm_test_cases = mllm_test_cases

    def add_test_case(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    ):
        validate_test_case_type(test_case, subject="test cases")

        test_case._dataset_alias = self._alias
        test_case._dataset_id = self._id
        if isinstance(test_case, LLMTestCase):
            test_case._dataset_rank = len(self._llm_test_cases)
            self._llm_test_cases.append(test_case)
        elif isinstance(test_case, ConversationalTestCase):
            test_case._dataset_rank = len(self._conversational_test_cases)
            self._conversational_test_cases.append(test_case)
        elif isinstance(test_case, MLLMTestCase):
            test_case._dataset_rank = len(self._mllm_test_cases)
            self._mllm_test_cases.append(test_case)

    def __iter__(self):
        return iter(self.test_cases)

    def evaluate(self, metrics: List[BaseMetric]):
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
            additional_metadata,
        ) in zip(
            inputs,
            actual_outputs,
            expected_outputs,
            contexts,
            retrieval_contexts,
            additional_metadatas,
        ):
            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
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
            with open(file_path, "r") as file:
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

            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                )
            )

    def push(self, alias: str, overwrite: Optional[bool] = None):
        if len(self.test_cases) == 0 and len(self.goldens) == 0:
            raise ValueError(
                "Unable to push empty dataset to Confident AI, there must be at least one test case or golden in dataset"
            )
        if is_confident():
            goldens = self.goldens
            goldens.extend(convert_test_cases_to_goldens(self.test_cases))
            api_dataset = APIDataset(
                alias=alias,
                overwrite=overwrite,
                goldens=goldens,
                conversationalGoldens=self.conversational_goldens,
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
                    "✅ Dataset successfully pushed to Confidnet AI! View at "
                    f"[link={link}]{link}[/link]"
                )
                webbrowser.open(link)
        else:
            raise Exception(
                "To push dataset to Confident AI, run `deepeval login`"
            )

    def pull(self, alias: str, auto_convert_goldens_to_test_cases: bool = True):
        if is_confident():
            api = Api()
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
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
                    params={"alias": alias},
                )

                conversational_goldens = []
                for cg in convert_keys_to_snake_case(
                    result["conversationalGoldens"]
                ):
                    if "goldens" in cg:
                        cg["turns"] = cg.pop("goldens")
                    conversational_goldens.append(ConversationalGolden(**cg))

                response = DatasetHttpResponse(
                    goldens=convert_keys_to_snake_case(result["goldens"]),
                    conversationalGoldens=conversational_goldens,
                    datasetId=result["datasetId"],
                )

                self._alias = alias
                self._id = response.datasetId

                if auto_convert_goldens_to_test_cases:
                    llm_test_cases = convert_goldens_to_test_cases(
                        response.goldens, alias, response.datasetId
                    )
                    conversational_test_cases = (
                        convert_convo_goldens_to_convo_test_cases(
                            response.conversational_goldens,
                            alias,
                            response.datasetId,
                        )
                    )
                    self._llm_test_cases.extend(llm_test_cases)
                    self._conversational_test_cases.extend(
                        conversational_test_cases
                    )
                else:
                    self.goldens = response.goldens
                    self.conversational_goldens = (
                        response.conversational_goldens
                    )

                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)",
                )
        else:
            raise Exception(
                "Run `deepeval login` to pull dataset from Confident AI"
            )

    def generate_red_teaming_goldens(
        self,
        synthesizer=None,
        contexts: Optional[List[List[str]]] = None,
        include_expected_output: bool = False,
        max_goldens: int = 2,
        num_evolutions: int = 1,
        attacks: List[RTAdversarialAttack] = [
            RTAdversarialAttack.PROMPT_INJECTION,
            RTAdversarialAttack.PROMPT_PROBING,
            RTAdversarialAttack.GRAY_BOX_ATTACK,
            RTAdversarialAttack.JAILBREAKING,
        ],
        vulnerabilities: List[RTVulnerability] = [
            RTVulnerability.BIAS,
            RTVulnerability.DATA_LEAKAGE,
            RTVulnerability.HALLUCINATION,
            RTVulnerability.OFFENSIVE,
            RTVulnerability.UNFORMATTED,
        ],
        use_case: UseCase = UseCase.QA,
    ):
        from deepeval.synthesizer import Synthesizer

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        self.goldens.extend(
            synthesizer.generate_red_teaming_goldens(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens=max_goldens,
                num_evolutions=num_evolutions,
                attacks=attacks,
                vulnerabilities=vulnerabilities,
                use_case=use_case,
                _send_data=False,
            )
        )

    def generate_goldens(
        self,
        contexts: List[List[str]],
        include_expected_output: bool = False,
        max_goldens_per_context: int = 2,
        num_evolutions: int = 1,
        source_files: Optional[List[str]] = None,
        evolutions: List[Evolution] = [
            Evolution.REASONING,
            Evolution.MULTICONTEXT,
            Evolution.CONCRETIZING,
            Evolution.CONSTRAINED,
            Evolution.COMPARATIVE,
            Evolution.HYPOTHETICAL,
            Evolution.IN_BREADTH,
        ],
        use_case: UseCase = UseCase.QA,
        synthesizer=None,
    ):
        from deepeval.synthesizer import Synthesizer

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        self.goldens.extend(
            synthesizer.generate_goldens(
                contexts=contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context,
                num_evolutions=num_evolutions,
                source_files=source_files,
                evolutions=evolutions,
                use_case=use_case,
                _send_data=False,
            )
        )

    def generate_goldens_from_docs(
        self,
        document_paths: List[str],
        include_expected_output: bool = False,
        max_goldens_per_document: int = 5,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        num_evolutions: int = 1,
        evolutions: List[Evolution] = [
            Evolution.REASONING,
            Evolution.MULTICONTEXT,
            Evolution.CONCRETIZING,
            Evolution.CONSTRAINED,
            Evolution.COMPARATIVE,
            Evolution.HYPOTHETICAL,
            Evolution.IN_BREADTH,
        ],
        use_case: UseCase = UseCase.QA,
        synthesizer=None,
    ):
        from deepeval.synthesizer import Synthesizer

        if synthesizer is None:
            synthesizer = Synthesizer()
        else:
            assert isinstance(synthesizer, Synthesizer)

        self.goldens.extend(
            synthesizer.generate_goldens_from_docs(
                document_paths=document_paths,
                include_expected_output=include_expected_output,
                max_goldens_per_document=max_goldens_per_document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_evolutions=num_evolutions,
                evolutions=evolutions,
                use_case=use_case,
                _send_data=False,
            )
        )

    # TODO: add save test cases as well
    def save_as(self, file_type: str, directory: str) -> str:
        if file_type not in valid_file_types:
            raise ValueError(
                f"Invalid file type. Available file types to save as: {', '.join(type for type in valid_file_types)}"
            )

        if len(self.goldens) == 0:
            raise ValueError(
                f"No synthetic goldens found. Please generate goldens before attempting to save data as {file_type}"
            )

        new_filename = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f".{file_type}"
        )

        if not os.path.exists(directory):
            os.makedirs(directory)

        full_file_path = os.path.join(directory, new_filename)

        if file_type == "json":
            with open(full_file_path, "w") as file:
                json_data = [
                    {
                        "input": golden.input,
                        "actual_output": golden.actual_output,
                        "expected_output": golden.expected_output,
                        "context": golden.context,
                        "source_file": golden.source_file,
                    }
                    for golden in self.goldens
                ]
                json.dump(json_data, file, indent=4)

        elif file_type == "csv":
            with open(full_file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "input",
                        "actual_output",
                        "expected_output",
                        "context",
                        "source_file",
                    ]
                )
                for golden in self.goldens:
                    writer.writerow(
                        [
                            golden.input,
                            golden.actual_output,
                            golden.expected_output,
                            "|".join(golden.context),
                            golden.source_file,
                        ]
                    )

        print(f"Evaluation dataset saved at {full_file_path}!")
        return full_file_path
