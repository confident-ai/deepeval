from typing import Optional, List
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import re

from deepeval.prompt.api import PromptHttpResponse, PromptMessage, PromptType
from deepeval.utils import is_confident
from deepeval.confident.api import Api, Endpoints, HttpMethods


class Prompt:
    _prompt_version_id: Optional[str] = None
    _type: Optional[PromptType] = None

    def __init__(
        self,
        alias: Optional[str] = None,
        template: Optional[str] = None,
        message_templates: Optional[List[PromptMessage]] = None,
    ):
        if alias is None and template is None:
            raise TypeError(
                "Unable to create Prompt where 'alias' and 'template' are both None. Please provide at least one to continue."
            )

        self.alias = alias
        self.template = template
        self.message_templates = message_templates
        self.version = None

    def interpolate(self, **kwargs):
        if self._type == PromptType.TEXT:
            if self.template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI or set template manually to continue."
                )

            formatted_template = re.sub(
                r"\{\{ (\w+) \}\}", r"{\1}", self.template
            )
            return formatted_template.format(**kwargs)
        elif self._type == PromptType.LIST:
            if self.message_templates is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template messages. Please pull a prompt from Confident AI or set template manually to continue."
                )

            interpolated_messages = []
            for message in self.message_templates:
                formatted_content = re.sub(
                    r"\{\{ (\w+) \}\}", r"{\1}", message.content
                )
                interpolated_content = formatted_content.format(**kwargs)
                interpolated_messages.append(
                    {"role": message.role, "content": interpolated_content}
                )
            return interpolated_messages
        else:
            raise ValueError(f"Unsupported prompt type: {self._type}")

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
                    promptVersionId=result["promptVersionId"],
                    template=result["value"],
                    messages=result["messages"],
                    type=result["type"],
                )
                self.version = version
                self.template = response.template
                self.message_templates = response.messages
                self._prompt_version_id = response.promptVersionId
                self._type = response.type

                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Done! ({time_taken}s)",
                )
        else:
            raise Exception(
                "Run `deepeval login` to pull prompt template from Confident AI"
            )
