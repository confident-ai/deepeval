"""Class for Evaluation Datasets
"""
import json
import random
import time
from tabulate import tabulate
from datetime import datetime
from typing import List, Callable
from collections import UserList
from deepeval.test_case import TestCase
from deepeval.metrics.metric import Metric
from deepeval.retry import retry


class EvaluationDataset(UserList):
    """Class for Evaluation Datasets -  which are a list of test cases"""

    def __init__(self, test_cases: List[TestCase]):
        self.data: List[TestCase] = test_cases

    @classmethod
    def from_csv(
        cls,  # Use 'cls' instead of 'self' for class methods
        csv_filename: str,
        query_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        import pandas as pd

        df = pd.read_csv(csv_filename)
        querys = df[query_column].values
        expected_outputs = df[expected_output_column].values
        if id_column is not None:
            ids = df[id_column].values

        # Initialize the 'data' attribute as an empty list
        cls.data = []

        for i, query_data in enumerate(querys):
            cls.data.append(
                TestCase(
                    query=query_data,
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
        query_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        """
        This is for JSON data in the format of key-value array pairs.
        {
            "query": ["What is the customer success number", "What is the customer success number"],
        }

        if the JSON data is in a list of dicionaries, use from_json_list
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []

        for i, query in enumerate(data[query_column]):
            test_cases.append(
                TestCase(
                    query=data[query_column][i],
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
        query_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        """
        This is for JSON data in the format of a list of dictionaries.
        [
            {"query": "What is the customer success number", "expected_output": "What is the customer success number"},
        ]
        """
        with open(json_filename, "r") as f:
            data = json.load(f)
        test_cases = []
        for i, query in enumerate(data):
            test_cases.append(
                TestCase(
                    query=data[i][query_column],
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
        query_key: str,
        expected_output_key: str,
        id_key: str = None,
        metrics: List[Metric] = None,
    ):
        pass

    def to_dict(self):
        return [x.dict() for x in self.data]

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
        return [r.dict() for r in result]

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]

    def run_evaluation(
        self,
        completion_fn: Callable,
        test_filename: str = None,
        max_retries: int = 3,
        min_success: int = 1,
        raise_error_on_run: bool = False,
        metrics: List[TestCase] = None,
    ):
        """Run evaluation with given metrics"""
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
            output = completion_fn(case.query)
            if metrics is None:
                metrics = case.metrics
            for metric in metrics:

                @retry(max_retries=max_retries, min_success=min_success)
                def assert_metric():
                    score = metric(output, case.expected_output)
                    is_successful: bool = metric.is_successful()
                    is_successful: bool = bool(is_successful)
                    message = f"""{metric.__class__.__name__} was unsuccessful for 
{case.query} 
which should have matched 
{case.expected_output}
"""
                    table.append(
                        [
                            is_successful,
                            metric.__class__.__name__,
                            score,
                            output,
                            case.expected_output,
                            message,
                        ]
                    )
                    assert is_successful, metric.__name__ + " wasn't successful"

                if raise_error_on_run:
                    assert_metric()
                else:
                    try:
                        assert_metric()
                    except AssertionError as e:
                        print(e)
                    except Exception as e:
                        print(e)
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
            from dash import Dash, dash_table, dcc, html, Input, Output, State, callback
        except ModuleNotFoundError:
            raise Exception(
                """You will need to run `pip install dash` to be able to review tests that were automatically created."""
            )

        table_data = [
            {"query": x.query, "expected_output": x.expected_output} for x in self.data
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
                            for i, c in enumerate(["query", "expected_output"])
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
                    writer = csv.DictWriter(f, fieldnames=[c["id"] for c in columns])
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
        """Utility function to create an evaluation dataset using GPT."""
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

Respond in JSON format in 1 single line without white spaces an array of JSON with the keys `query` and `answer`.
"""
    for _ in range(3):
        try:
            responses = generate_chatgpt_output(prompt, openai_api_key=openai_api_key)
            responses = json.loads(responses)
            break
        except Exception as e:
            return EvaluationDataset(test_cases=[])

    test_cases = []
    for response in responses:
        test_case = TestCase(
            query=response["query"], expected_output=response["answer"]
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset
