from typing import List
from rich import print
import typer

from deepeval.cli.types import (
    RecommendMetricsRequestData,
    RecommendMetricsResponseData,
)
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.telemetry import capture_recommend_metrics
from deepeval.constants import LOGIN_PROMPT

app = typer.Typer(name="recommend")


def get_next_question(question_index: int):
    recommend_metrics_request_data = RecommendMetricsRequestData(
        questionIndex=question_index,
        userAnswers=None,
    )
    body = recommend_metrics_request_data.model_dump(exclude_none=True)
    api = Api(api_key="NA")
    response = api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.RECOMMEND_ENDPOINT,
        body=body,
    )
    return RecommendMetricsResponseData(**response)


def get_recommended_metrics(question_index: int, user_answers: List[bool]):
    recommend_metrics_request_data = RecommendMetricsRequestData(
        questionIndex=question_index,
        userAnswers=user_answers,
    )
    body = recommend_metrics_request_data.model_dump(exclude_none=True)
    api = Api(api_key="NA")
    response = api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.RECOMMEND_ENDPOINT,
        body=body,
    )
    return RecommendMetricsResponseData(**response)


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/N]").strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        else:
            print(
                "[red]Invalid input.[/red] Please enter '[rgb(5,245,141)]Y[/rgb(5,245,141)]' for Yes or '[rgb(5,245,141)]N[/rgb(5,245,141)]' for No."
            )


@app.command()
def metrics():
    with capture_recommend_metrics() as span:
        try:
            print(
                "\n[bold]Welcome to [cyan]DeepEval[/cyan]! Let's find the best evaluation metrics for you.[/bold] :sparkles:\n"
            )
            print(f"{LOGIN_PROMPT}\n")

            is_last_question = False
            question_index = 0
            user_answers = []

            while True:
                response: RecommendMetricsResponseData = get_next_question(
                    question_index
                )
                question = response.question
                is_last_question = response.isLastQuestion

                if question:
                    answer = ask_yes_no(question)
                    user_answers.append(answer)

                if is_last_question:
                    print(
                        "\n[bold rgb(5,245,141)]:rocket: Generating your recommended metrics...[/bold rgb(5,245,141)]\n"
                    )
                    response: RecommendMetricsResponseData = (
                        get_recommended_metrics(
                            question_index + 1, user_answers
                        )
                    )

                    print("[bold cyan]Recommended Metrics:[/bold cyan]")
                    for metric in response.recommendedMetrics:
                        print(f" -  {metric}")

                    print(
                        "\n:clap: [bold]You're all set![/bold] You can also run '[bold cyan]deepeval login[/bold cyan]' to get reports of your metric scores on Confident AI.\n"
                    )
                    break

                question_index += 1
        except:
            span.set_attribute("completed", False)
