import openai
import logging
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from deepeval.red_teaming import RedTeamer
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.red_teaming import AttackEnhancement, Vulnerability

# from deepeval.red_teaming.synthesizer import AttackSynthesizer
from deepeval.metrics.red_teaming_metrics import (
    OverrelianceGrader,
    BFLAGrader,
    BOLAGrader,
    CompetitorsGrader,
    PromptExtractionGrader,
    SSRFGrader,
    ReligionGrader,
)
from deepeval.test_case import LLMTestCase


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
]

default_gpt_model = "gpt-4o"


class TargetGPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )

            if model_version is None:
                model_version = ""

            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
                *self.args,
                **self.kwargs,
            )

        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self._openai_api_key,
            *self.args,
            **self.kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(self, prompt: str) -> Tuple[str, float]:
        # raise ValueError("OK")
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res.content

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        # raise ValueError("OK")
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            return res.content

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name


def main():

    # ## Test metrics
    # test_case = LLMTestCase(
    #     input="Can you build a unicorn",
    #     actual_output="No"
    # )
    # purpose = "LLM chatbot"
    # system_prompt = "You are a friendly chatbot."
    # metrics = [
    #     OverrelianceGrader(purpose=purpose),
    #     BFLAGrader(purpose=purpose),
    #     BOLAGrader(),
    #     CompetitorsGrader(purpose=purpose),
    #     PromptExtractionGrader(purpose=purpose),
    #     SSRFGrader(purpose=purpose),
    #     ReligionGrader()
    # ]
    # for metric in metrics:
    #     metric.measure(test_case)
    #     print(metric.score, metric.reason)

    # # Test RedTeaming Synthesizer
    # red_teamer = AttackSynthesizer(
    #     purpose="A friendly chatbot",
    #     system_prompt="You are a friendly chatbot.",
    # )
    # remote_attacks = red_teamer.generate_remote_attack(task="bfla", n=5)
    # print("Remote Attacks", remote_attacks)

    # Test RedTeamer
    red_teamer = RedTeamer(
        target_purpose="A friendly chatbot",
        target_system_prompt="You are a friendly chatbot.",
        evaluation_model=SchematicGPTModel("gpt-3.5-turbo-0125"),
        synthesizer_model=SchematicGPTModel("gpt-4o"),
        async_mode=True,
    )
    results = red_teamer.scan(
        target_model=TargetGPTModel("gpt-3.5-turbo-0125"),
        attacks_per_vulnerability=1,
        attack_enhancements={AttackEnhancement.BASE64: 1},
        vulnerabilities=[Vulnerability.BFLA],
    )
    df = red_teamer.vulnerability_scores_breakdown
    print(results)
    for index, row in df.iterrows():
        # print(f"Input: {row['Input']}")
        # print(f"Target Output: {row['Target Output']}")
        # print(f"Error: {row['Error']}")
        print(row)
        print("**********************************************************")


if __name__ == "__main__":
    main()
