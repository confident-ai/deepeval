"""Class for Evaluation Datasets
"""
import json
import random
from tabulate import tabulate
from datetime import datetime
from typing import List, Callable
from collections import UserList
from .test_case import TestCase
from .metrics.metric import Metric
from .query_generator import BEIRQueryGenerator


class EvaluationDataset(UserList):
    """Class for Evaluation Datasets -  which are a list of test cases"""

    def __init__(self, test_cases: List[TestCase]):
        self.data = test_cases

    @classmethod
    def from_csv(
        cls,  # Use 'cls' instead of 'self' for class methods
        csv_filename: str,
        input_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        import pandas as pd

        df = pd.read_csv(csv_filename)
        inputs = df[input_column].values
        expected_outputs = df[expected_output_column].values
        if id_column is not None:
            ids = df[id_column].values

        # Initialize the 'data' attribute as an empty list
        cls.data = []

        for i, input_data in enumerate(inputs):
            cls.data.append(
                TestCase(
                    input=input_data,
                    expected_output=expected_outputs[i],
                    metrics=metrics,
                    id=ids[i] if id_column else None,
                )
            )
        return cls(cls.data)

    def from_test_cases(self, test_cases: list):
        self.data = test_cases

    @classmethod
    def from_json(
        cls,
        json_filename: str,
        input_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        """
        This is for JSON data in the format of key-value array pairs.
        {
            "input": ["What is the customer success number", "What is the customer success number"],
        }

        if the JSON data is in a list of dicionaries, use from_json_list
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []

        for i, input in enumerate(data[input_column]):
            test_cases.append(
                TestCase(
                    input=data[input_column][i],
                    expected_output=data[expected_output_column][i],
                    metrics=metrics,
                    id=data[id_column][i],
                )
            )
        return cls(data)

    @classmethod
    def from_json_list(
        cls,
        json_filename: str,
        input_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        """
        This is for JSON data in the format of a list of dictionaries.
        [
            {"input": "What is the customer success number", "expected_output": "What is the customer success number"},
        ]
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []
        for i, input in enumerate(data):
            test_cases.append(
                TestCase(
                    input=data[i][input_column],
                    expected_output=data[i][expected_output_column],
                    metrics=metrics,
                    id=data[i][id_column],
                )
            )
        return cls(data)

    @classmethod
    def from_dict(
        cls,
        data: List[dict],
        input_key: str,
        expected_output_key: str,
        id_key: str = None,
        metrics: List[Metric] = None,
    ):
        pass

    def to_csv(self, csv_filename: str):
        import pandas as pd

        df = pd.DataFrame(self.data)
        df.to_csv(csv_filename, index=False)

    def to_json(self, json_filename: str):
        with open(json_filename, "w") as f:
            json.dump(self.data, f)

    def from_hf_evals(self):
        raise NotImplementedError

    def from_df(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def sample(self, n: int = 5):
        return random.sample(self.data, n)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]

    def run_evaluation(self, completion_fn: Callable, test_filename: str = None):
        table = []

        headers = [
            "Test Passed",
            "Metric Name",
            "Score",
            "Output",
            "Expected output",
            "Message",
        ]
        for case in self.data:
            case: TestCase
            output = completion_fn(case.input)
            for metric in case.metrics:
                score = metric(output, case.expected_output)
                is_successful = metric.is_successful()
                message = f"""{metric.__class__.__name__} was unsuccessful for 
{case.input} 
which should have matched 
{case.expected_output}
"""
                table.append(
                    [
                        bool(is_successful),
                        metric.__class__.__name__,
                        score,
                        output,
                        case.expected_output,
                        message,
                    ]
                )
        if test_filename is None:
            test_filename = (
                f"test-result-{datetime.now().__str__().replace(' ', '-')}.txt"
            )
        with open(test_filename, "w") as f:
            f.write(tabulate(table, headers=headers))
        print(f"Saved to {test_filename}")
        for t in table:
            assert t[0] == True, t[-1]
        return test_filename


def create_evaluation_dataset_from_raw_text(text: str, output_fn: str = "output.csv"):
    """Utility function to create an evaluation dataset from raw text."""
    print(f"Outputting to {output_fn}")

    # NOTE: loading this may take a while as the model used is quite big
    gen = BEIRQueryGenerator()
    text = "Synthetic queries are useful for scenraios where there is no data."
    queries = gen.generate_queries(texts=[text], num_queries=2)
    test_cases = []
    with open(output_fn, "w") as f:
        f.write("input,expected_output\n")
        for query in queries:
            f.write(f"{query}, {text}\n")
        test_case = TestCase(input=text, expected_output=text)
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset
