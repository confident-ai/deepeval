from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
from rich.console import Console
import json
import webbrowser
import os

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluator import evaluate
from deepeval.api import Api, Endpoints
from deepeval.dataset.utils import convert_test_cases_to_goldens
from deepeval.dataset.api import APIDataset, CreateDatasetHttpResponse


@dataclass
class EvaluationDataset:
    test_cases: List[LLMTestCase]

    def __init__(self, test_cases: List[LLMTestCase] = []):
        self.test_cases = test_cases

    def add_test_case(self, test_case: LLMTestCase):
        self.test_cases.append(test_case)

    def __iter__(self):
        return iter(self.test_cases)

    def evaluate(self, metrics: List[BaseMetric]):
        return evaluate(self.test_cases, metrics)

    def add_test_cases_from_csv_file(
        self,
        file_path: str,
        input_col_name: str,
        actual_output_col_name: str,
        expected_output_col_name: Optional[str] = None,
        context_col_name: Optional[str] = None,
        context_col_delimiter: str = ";",
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

        Returns:
            None: The method adds test cases to the Dataset instance but does not return anything.

        Raises:
            FileNotFoundError: If the CSV file specified by `file_path` cannot be found.
            pd.errors.EmptyDataError: If the CSV file is empty.
            KeyError: If one or more specified columns are not found in the CSV file.

        Note:
            The CSV file is expected to contain columns as specified in the arguments. Each row in the file represents a single test case. The method assumes the file is properly formatted and the specified columns exist. For context data represented as lists in the CSV, ensure the correct delimiter is specified.
        """
        df = pd.read_csv(file_path)

        inputs = self._get_column_data(df, input_col_name)
        actual_outputs = self._get_column_data(df, actual_output_col_name)
        expected_outputs = self._get_column_data(
            df, expected_output_col_name, default=None
        )
        contexts = [
            context.split(context_col_delimiter) if context else []
            for context in self._get_column_data(
                df, context_col_name, default=""
            )
        ]

        for input, actual_output, expected_output, context in zip(
            inputs, actual_outputs, expected_outputs, contexts
        ):
            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                )
            )

    def _get_column_data(self, df: pd.DataFrame, col_name: str, default=None):
        return (
            df[col_name].values
            if col_name in df.columns
            else [default] * len(df)
        )

    def add_test_cases_from_json_file(
        self,
        file_path: str,
        input_key_name: str,
        actual_output_key_name: str,
        expected_output_key_name: Optional[str] = None,
        context_key_name: Optional[str] = None,
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

            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                )
            )

    def add_test_cases_from_hf_dataset(
        self,
        dataset_name: str,
        input_field_name: str,
        actual_output_field_name: str,
        expected_output_field_name: Optional[str] = None,
        context_field_name: Optional[str] = None,
        split: str = "train",
    ):
        """
        Load test cases from a Hugging Face dataset.

        This method loads a specified dataset and split from Hugging Face's datasets library, then iterates through each entry to create and add LLMTestCase objects to the Dataset instance based on specified field names.

        Args:
            dataset_name (str): The name of the Hugging Face dataset to load.
            split (str): The split of the dataset to load (e.g., 'train', 'test', 'validation'). Defaults to 'train'.
            input_field_name (str): The field name in the dataset corresponding to the input for the test case.
            actual_output_field_name (str): The field name in the dataset corresponding to the actual output for the test case.
            expected_output_field_name (str, optional): The field name in the dataset corresponding to the expected output for the test case. Defaults to None.
            context_field_name (str, optional): The field name in the dataset corresponding to the context for the test case. Defaults to None.

        Returns:
            None: The method adds test cases to the Dataset instance but does not return anything.

        Raises:
            ValueError: If the required fields (input and actual output) are not found in the dataset.
            FileNotFoundError: If the specified dataset is not available in Hugging Face's datasets library.
            datasets.DatasetNotFoundError: Specific Hugging Face error if the dataset or split is not found.
            json.JSONDecodeError: If there is an issue in reading or processing the dataset.

        Note:
            Ensure that the dataset structure aligns with the expected field names. The method assumes each dataset entry is a dictionary-like object.
        """

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is missing. Please install it using pip: pip install datasets"
            )
        hf_dataset = load_dataset(dataset_name, split=split)

        # Process each entry in the dataset
        for entry in hf_dataset:
            if (
                input_field_name not in entry
                or actual_output_field_name not in entry
            ):
                raise ValueError(
                    "Required fields are missing in one or more dataset entries"
                )

            input = entry[input_field_name]
            actual_output = entry[actual_output_field_name]
            expected_output = entry.get(expected_output_field_name)
            context = entry.get(context_field_name)

            self.add_test_case(
                LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                )
            )

    def push(self, alias: str):
        if len(self.test_cases) == 0:
            raise ValueError(
                "Unable to push empty dataset to Confident AI, there must be at least one test case in dataset"
            )
        if os.path.exists(".deepeval"):
            goldens = convert_test_cases_to_goldens(self.test_cases)
            body = APIDataset(alias=alias, goldens=goldens).model_dump(
                by_alias=True, exclude_none=True
            )
            api = Api()
            result = api.post_request(
                endpoint=Endpoints.CREATE_DATASET_ENDPOINT.value,
                body=body,
            )
            response = CreateDatasetHttpResponse(
                link=result["link"],
            )
            link = response.link
            console = Console()
            console.print(
                "✅ Dataset pushed to Confidnet AI! View on "
                f"[link={link}]{link}[/link]"
            )
            # webbrowser.open(link)
        else:
            raise Exception(
                "To push dataset to Confident AI, run `deepeval login`"
            )

    # TODO
    def pull(self, alias: str):
        pass
