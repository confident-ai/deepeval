from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import re

from deepeval.prompt.api import PromptHttpResponse
from deepeval.utils import is_confident
from deepeval.confident.api import Api, Endpoints, HttpMethods


class Prompt:
    def __init__(self, alias: str):
        self.alias = alias
        self.template = None
        self.version = None

    def interpolate(self, **kwargs):
        if self.template is None:
            raise TypeError(
                "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI to continue."
            )

        formatted_template = re.sub(r"\{\{ (\w+) \}\}", r"{\1}", self.template)
        return formatted_template.format(**kwargs)

    def pull(self, version: Optional[str] = None):
        if self.alias is None:
            raise TypeError(
                "Unable to pull prompt from Confident AI when no alias is provided."
            )

        if is_confident():
            api = Api()
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Pulling [rgb(106,0,255)]'{self.alias}' (version='{version or 'latest'}')[/rgb(106,0,255)] from Confident AI...",
                    total=100,
                )
                start_time = time.perf_counter()
                result = api.send_request(
                    method=HttpMethods.GET,
                    endpoint=Endpoints.PROMPT_ENDPOINT,
                    params={"alias": self.alias, "version": version},
                )
                response = PromptHttpResponse(
                    value=result["value"], version=result["version"]
                )
                self.template = response.value
                self.version = response.version

                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Done! ({time_taken}s)",
                )
        else:
            raise Exception(
                "Run `deepeval login` to pull prompt from Confident AI"
            )
