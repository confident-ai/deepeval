from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel
# import httpx
# import json

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json

# check boto availability
try:
    import boto3
    # from botocore.awsrequest import AWSRequest
    # from botocore.auth import SigV4Auth
    from botocore.config import Config
    boto_available = True
except ImportError:
    boto_available = False

def _check_boto_available():
    if not boto_available:
        raise ImportError(
            "boto3 and botocore are required for this functionality. Install it via your package manager"
        )

class AmazonBedrockModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_id: str,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        temperature: float = 0,
        input_token_cost: float = 0,
        output_token_cost: float = 0
    ):
        _check_boto_available()
        self.model_id = model_id
        self.temperature = temperature
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.top_p = 0
        self.max_tokens = 10000

        if self.temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        super().__init__(model_id)

        # TODO: Potentially reimplement a_generate once aiobotocore is compatible with boto3
        # boto_sess = boto3.Session(
        #     aws_access_key_id=aws_access_key_id,
        #     aws_secret_access_key=aws_secret_access_key,
        #     region_name=region_name,
        # )
        # creds = boto_sess.get_credentials().get_frozen_credentials()
        # self._signer = SigV4Auth(creds, "bedrock", region_name)
        # base = f"https://bedrock-runtime.{region_name}.amazonaws.com"
        # self._conv_url = f"{base}/model/{model_id}/converse"
        # self._http = httpx.AsyncClient(timeout=None)

    ###############################################
    # Generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:     
        converse_request = self.get_converse_request_body(prompt)
        client = self.load_model()
        response = client.converse(
            modelId=self.model_id,
            messages=converse_request["messages"],
            inferenceConfig=converse_request["inferenceConfig"]
        )
        message = response["output"]["message"]["content"][0]["text"]
        cost = self.calculate_cost(response["usage"]["inputTokens"], response["usage"]["outputTokens"])
        if schema is None:
            return message, cost
        else:
        # TODO: Decide whether to enforce JSON output via Instructor (sacrifice cost reporting)
            json_output = trim_and_load_json(message)
            return schema.model_validate(json_output), cost

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:     
        converse_request = self.get_converse_request_body(prompt)
        client = self.load_model()
        response = client.converse(
            modelId=self.model_id,
            messages=converse_request["messages"],
            inferenceConfig=converse_request["inferenceConfig"]
        )
        message = response["output"]["message"]["content"][0]["text"]
        cost = self.calculate_cost(response["usage"]["inputTokens"], response["usage"]["outputTokens"])
        if schema is None:
            return message, cost
        else:
            json_output = trim_and_load_json(message)
            return schema.model_validate(json_output), cost

    ###############################################
    # Helper Functions
    ###############################################
    
    # TODO: Potentially reimplement a_generate once aiobotocore is compatible with boto3
    #  async def a_generate(
    #     self, prompt: str, schema: Optional[BaseModel] = None
    # ) -> Tuple[Union[str, Dict], float]:
    #     converse_request = self.get_converse_request_body(prompt)
    #     response = await self.a_converse(converse_request)
    #     print(response)
    #     message = response["output"]["message"]["content"][0]["text"]
    #     cost = self.calculate_cost(response["usage"]["inputTokens"], response["usage"]["outputTokens"])
    #     if schema is None:
    #         return message, cost
    #     else:
    #         json_output = trim_and_load_json(message)
    #         return schema.model_validate(json_output), cost

    # async def a_converse(self, payload: dict) -> Dict:
    #     body = json.dumps(payload)
    #     aws_req = AWSRequest(
    #         method="POST",
    #         url=self._conv_url,
    #         data=body,
    #         headers={"Content-Type": "application/json"},
    #     )
    #     self._signer.add_auth(aws_req)
    #     prepped = aws_req.prepare()
    #     resp = await self._http.post(
    #         self._conv_url,
    #         headers=dict(prepped.headers),
    #         content=prepped.body,
    #     )
    #     resp.raise_for_status()
    #     return resp.json()

    def get_converse_request_body(self, prompt):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxTokens": self.max_tokens
            }
        }

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens * self.input_token_cost
        output_cost = output_tokens * self.output_token_cost
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self):
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(
            retries={
                "max_attempts": 5,
                "mode": "adaptive"
            }
        )
        )
        return bedrock_client

    def get_model_name(self):
        return self.model_id
