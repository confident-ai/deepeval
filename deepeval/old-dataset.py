"""Class for Evaluation Datasets
"""
import json
import random
import time
from collections import UserList
from datetime import datetime
from typing import Any, Callable, List, Optional

from tabulate import tabulate

from deepeval.evaluator import run_test
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from dataclasses import asdict


class EvaluationDataset(UserList):
    """Class for Evaluation Datasets -  which are a list of test cases"""

    def __init__(self, test_cases: List[LLMTestCase]):
        self.data: List[LLMTestCase] = test_cases

    @classmethod
    def from_csv(
        cls,  # Use 'cls' instead of 'self' for class methods
        csv_filename: str,
        query_column: Optional[str] = None,
        expected_output_column: Optional[str] = None,
        context_column: Optional[str] = None,
        output_column: Optional[str] = None,
        id_column: str = None,
        metrics: List[BaseMetric] = None,
    ):
        import pandas as pd

        df = pd.read_csv(csv_filename)
        if query_column is not None and query_column in df.columns:
            querys = df[query_column].values
        else:
            querys = [None] * len(df)
        if (
            expected_output_column is not None
            and expected_output_column in df.columns
        ):
            expected_outputs = df[expected_output_column].values
        else:
            expected_outputs = [None] * len(df)
        if context_column is not None and context_column in df.columns:
            contexts = df[context_column].values
        else:
            contexts = [None] * len(df)
        if output_column is not None and output_column in df.columns:
            outputs = df[output_column].values
        else:
            outputs = [None] * len(df)
        if id_column is not None:
            ids = df[id_column].values
        else:
            ids = [None] * len(df)

        # Initialize the 'data' attribute as an empty list
        cls.data = []

        for i, query_data in enumerate(querys):
            cls.data.append(
                LLMTestCase(
                    input=query_data,
                    expected_output=expected_outputs[i],
                    context=contexts[i],
                    id=ids[i] if id_column else None,
                    actual_output=outputs[i] if output_column else None,
                )
            )
        return cls(cls.data)

    def from_test_cases(self, test_cases: list):
        self.data = test_cases

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        split: str,
        query_column: str,
        expected_output_column: str,
        context_column: str = None,
        output_column: str = None,
        id_column: str = None,
    ):
        """
        Load test cases from a HuggingFace dataset.

        Args:
            dataset_name (str): The name of the HuggingFace dataset to load.
            split (str): The split of the dataset to load (e.g., 'train', 'test').
            query_column (str): The column in the dataset corresponding to the query.
            expected_output_column (str): The column in the dataset corresponding to the expected output.
            context_column (str, optional): The column in the dataset corresponding to the context. Defaults to None.
            output_column (str, optional): The column in the dataset corresponding to the output. Defaults to None.
            id_column (str, optional): The column in the dataset corresponding to the ID. Defaults to None.

        Returns:
            EvaluationDataset: An instance of EvaluationDataset containing the loaded test cases.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is missing. Please install it using pip: pip install datasets"
            )

        hf_dataset = load_dataset(dataset_name, split=split)
        test_cases = []

        for i, row in enumerate(hf_dataset):
            test_cases.append(
                LLMTestCase(
                    input=row[query_column],
                    expected_output=row[expected_output_column],
                    context=row[context_column] if context_column else None,
                    actual_output=row[output_column] if output_column else None,
                    id=row[id_column] if id_column else None,
                )
            )
        return cls(test_cases)

    @classmethod
    def from_json(
        cls,
        json_filename: str,
        query_column: str,
        expected_output_column: str,
        context_column: str,
        output_column: str,
        id_column: str = None,
    ):
        """
        This is for JSON data in the format of key-value array pairs.
        {
            "query": ["What is the customer success number", "What is the customer success number"],
            "context": ["Context 1", "Context 2"],
            "output": ["Output 1", "Output 2"]
        }

        if the JSON data is in a list of dictionaries, use from_json_list
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []

        for i, query in enumerate(data[query_column]):
            test_cases.append(
                LLMTestCase(
                    input=data[query_column][i],
                    expected_output=data[expected_output_column][i],
                    context=data[context_column][i],
                    actual_output=data[output_column][i],
                    id=data[id_column][i] if id_column else None,
                )
            )
        return cls(test_cases)

    @classmethod
    def from_json_list(
        cls,
        json_filename: str,
        query_column: str,
        expected_output_column: str,
        context_column: str,
        output_column: str,
        id_column: str = None,
    ):
        """
        This is for JSON data in the format of a list of dictionaries.
        [
            {"query": "What is the customer success number", "expected_output": "What is the customer success number", "context": "Context 1", "output": "Output 1"},
        ]
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []
        for i, query in enumerate(data):
            test_cases.append(
                LLMTestCase(
                    input=data[i][query_column],
                    expected_output=data[i][expected_output_column],
                    context=data[i][context_column],
                    actual_output=data[i][output_column],
                    id=data[i][id_column] if id_column else None,
                )
            )
        return cls(test_cases)

    @classmethod
    def from_dict(
        cls,
        data: List[dict],
        query_key: str,
        expected_output_key: str,
        context_key: str = None,
        output_key: str = None,
        id_key: str = None,
    ):
        """
        Load test cases from a list of dictionaries.

        Args:
            data (List[dict]): The list of dictionaries containing the test case data.
            query_key (str): The key in each dictionary corresponding to the query.
            expected_output_key (str): The key in each dictionary corresponding to the expected output.
            context_key (str, optional): The key in each dictionary corresponding to the context. Defaults to None.
            output_key (str, optional): The key in each dictionary corresponding to the output. Defaults to None.
            id_key (str, optional): The key in each dictionary corresponding to the ID. Defaults to None.
            metrics (List[BaseMetric], optional): The list of metrics to be associated with the test cases. Defaults to None.

        Returns:
            EvaluationDataset: An instance of EvaluationDataset containing the loaded test cases.
        """
        test_cases = []
        for i, case_data in enumerate(data):
            test_cases.append(
                LLMTestCase(
                    input=case_data[query_key],
                    expected_output=case_data[expected_output_key],
                    context=case_data[context_key] if context_key else None,
                    actual_output=case_data[output_key] if output_key else None,
                    id=case_data[id_key] if id_key else None,
                )
            )
        return cls(test_cases)

    def to_dict(self):
        return [asdict(x) for x in self.data]

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
        if len(self.data) <= n:
            n = len(self.data)
        result = random.sample(self.data, n)
        return [asdict(r) for r in result]

    def head(self, n: int = 5):
        return self.data[:n]

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]

    def run_evaluation(
        self,
        completion_fn: Callable[[str], str] = None,
        outputs: List[str] = None,
        test_filename: str = None,
        max_retries: int = 3,
        min_success: int = 1,
        metrics: List[BaseMetric] = None,
    ) -> str:
        """Run evaluation with given metrics"""
        if completion_fn is None:
            assert outputs is not None

        table: List[List[Any]] = []

        headers: List[str] = [
            "Test Passed",
            "Metric Name",
            "Score",
            "Output",
            "Expected output",
            "Message",
        ]
        results = run_test(
            test_cases=self.data,
            metrics=metrics,
            raise_error=True,
            max_retries=max_retries,
            min_success=min_success,
        )
        for result in results:
            table.append(
                [
                    result.success,
                    result.metric_name,
                    result.score,
                    result.output,
                    result.expected_output,
                    "",
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

    def review(self):
        """A bulk editor for reviewing synthetic data."""
        try:
            from dash import (
                Dash,
                Input,
                Output,
                State,
                callback,
                dash_table,
                dcc,
                html,
            )
        except ModuleNotFoundError:
            raise Exception(
                """You will need to run `pip install dash` to be able to review tests that were automatically created."""
            )

        table_data = [
            {"input": x.query, "expected_output": x.expected_output}
            for x in self.data
        ]
        app = Dash(
            __name__,
            external_stylesheets=[
                "https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/darkly/bootstrap.min.css"
            ],
        )

        app.layout = html.Div(
            [
                html.H1("Bulk Review Test Cases", style={"marginLeft": "20px"}),
                html.Button(
                    "Add Test case",
                    id="editing-rows-button",
                    n_clicks=0,
                    style={
                        "padding": "8px",
                        "backgroundColor": "purple",  # Added purple background color
                        "color": "white",
                        "border": "2px solid purple",  # Added purple border
                        "marginLeft": "20px",
                    },
                ),
                html.Div(
                    dash_table.DataTable(
                        id="adding-rows-table",
                        columns=[
                            {
                                "name": c.title().replace("_", " "),
                                "id": c,
                                "deletable": True,
                                "renamable": True,
                            }
                            for i, c in enumerate(["input", "expected_output"])
                        ],
                        data=table_data,
                        editable=True,
                        row_deletable=True,
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "rgb(40, 40, 40)",
                                "color": "white",
                            },
                            {
                                "if": {"row_index": "even"},
                                "backgroundColor": "rgb(30, 30, 30)",
                                "color": "white",
                            },
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "white",
                                "color": "white",
                            },
                        ],
                        style_header={
                            "backgroundColor": "rgb(30, 30, 30)",
                            "color": "white",
                            "fontWeight": "bold",
                            "padding": "10px",  # Added padding
                        },
                        style_cell={
                            "padding": "10px",  # Added padding
                            "whiteSpace": "pre-wrap",  # Wrap cell contents
                            "maxHeight": "200px",
                        },
                    ),
                    style={"padding": "20px"},  # Added padding
                ),
                html.Div(style={"margin-top": "20px"}),
                html.Button(
                    "Save To CSV",
                    id="save-button",
                    n_clicks=0,
                    style={
                        "padding": "8px",
                        "backgroundColor": "purple",  # Added purple background color
                        "color": "white",
                        "border": "2px solid purple",  # Added purple border
                        "marginLeft": "20px",
                    },
                ),
                dcc.Input(
                    id="filename-input",
                    type="text",
                    placeholder="Enter filename (.csv format)",
                    style={
                        "padding": "8px",
                        "backgroundColor": "rgb(30, 30, 30)",
                        "color": "white",
                        "marginLeft": "20px",
                        "border": "2px solid purple",  # Added purple border
                        "width": "200px",  # Edited width
                    },
                    value="review-test.csv",
                ),
                html.Div(id="code-output"),
            ],
            style={"padding": "20px"},  # Added padding
        )

        @callback(
            Output("adding-rows-table", "data"),
            Input("editing-rows-button", "n_clicks"),
            State("adding-rows-table", "data"),
            State("adding-rows-table", "columns"),
        )
        def add_row(n_clicks, rows, columns):
            if n_clicks > 0:
                rows.append({c["id"]: "" for c in columns})
            return rows

        @callback(
            Output("save-button", "n_clicks"),
            Input("save-button", "n_clicks"),
            State("adding-rows-table", "data"),
            State("adding-rows-table", "columns"),
            State("filename-input", "value"),
        )
        def save_data(n_clicks, rows, columns, filename):
            if n_clicks > 0:
                import csv

                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=[c["id"] for c in columns]
                    )
                    writer.writeheader()
                    writer.writerows(rows)
            return n_clicks

        @app.callback(
            Output("code-output", "children"),
            Input("save-button", "n_clicks"),
            State("filename-input", "value"),
        )
        def show_code(n_clicks, filename):
            if n_clicks > 0:
                code = f"""
        from deepeval.dataset import EvaluationDataset

        # Replace 'filename.csv' with the actual filename
        ds = EvaluationDataset.from_csv('{filename}')

        # Access the data in the CSV file
        # For example, you can print a few rows
        print(ds.sample())
        """
                return html.Div(
                    [
                        html.P(
                            "Code to load the CSV file back into a dataset for testing:"
                        ),
                        html.Pre(code, className="language-python"),
                    ],
                    style={"padding": "20px"},  # Added padding
                )
            else:
                return ""

        app.run(debug=False)

    def add_evaluation_query_answer_pairs(
        self,
        openai_api_key: str,
        context: str,
        n: int = 3,
        model: str = "openai/gpt-3.5-turbo",
    ):
        """Utility function to create an evaluation dataset using ChatGPT."""
        new_dataset = create_evaluation_query_answer_pairs(
            openai_api_key=openai_api_key, context=context, n=n, model=model
        )
        self.data += new_dataset.data
        print(f"Added {len(new_dataset.data)}!")


def make_chat_completion_request(prompt: str, openai_api_key: str):
    import openai

    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def generate_chatgpt_output(prompt: str, openai_api_key: str) -> str:
    max_retries = 3
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            expected_output = make_chat_completion_request(
                prompt=prompt, openai_api_key=openai_api_key
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise

    return expected_output


def create_evaluation_query_answer_pairs(
    openai_api_key: str,
    context: str,
    n: int = 3,
    model: str = "openai/gpt-3.5-turbo",
) -> EvaluationDataset:
    """Utility function to create an evaluation dataset using GPT."""
    prompt = f"""You are generating {n} sets of of query-answer pairs to create an evaluation dataset based on the below context.
Context: {context}

Respond in JSON format in 1 single line without white spaces an array of JSON with the keys `query` and `answer`. Do not use any other keys in the response.
JSON:"""
    for _ in range(3):
        try:
            responses = generate_chatgpt_output(
                prompt, openai_api_key=openai_api_key
            )
            responses = json.loads(responses)
            break
        except Exception as e:
            print(e)
            return EvaluationDataset(test_cases=[])

    test_cases = []
    for response in responses:
        test_case = LLMTestCase(
            input=response["query"],
            expected_output=response["answer"],
            context=context,
            # store this as None for now
            actual_output="-",
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset
