import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import json
from pydantic import BaseModel, ValidationError
import logging
from typing import Union, Optional
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Update this for other regions
valid_bedrock_models = [
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "eu.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

default_bedrock_model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
default_multimodal_bedrock_model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
default_system_message = "You are a helpful AI assistant. Always generate your response as a valid json object without '```json'. No explanation or extra information is needed just the json."

class BedrockModel(DeepEvalBaseLLM):
    """A class that integrates with AWS Bedrock for model inference and text generation.

    This class communicates with the AWS Bedrock service to invoke models for generating text and extracting
    JSON responses from the model outputs.

    Attributes:
        model_id (str): The ID of the Bedrock model to use for inference.
        system_prompt (str): A predefined system prompt for Bedrock models that directs their behavior.
        access_key_id (str, optional): AWS access key ID for authentication. Can be provided or fetched from the key handler.
        secret_access_key (str, optional): AWS secret access key for authentication. Can be provided or fetched from the key handler.
        session_token (str, optional): AWS session token for temporary authentication. Can be provided or fetched from the key handler.
        region (str, optional): AWS region where the Bedrock client will be created. If not provided, defaults to fetched value.

    Example:
        ```python
        from deepeval.models import BedrockModel

        # Initialize the model with your own model ID and system prompt
        model = BedrockModel(
            model_id="your-bedrock-model-id",
            system_prompt="You are a helpful AI assistant. Always generate your response as a valid json. No explanation is needed just the json."
        )
        
        # Generate text with a prompt
        response = model.generate("What is the capital of France?", schema)
        ```
    """
    def __init__(
        self,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initializes the BedrockModel with model_id, system_prompt, and optional AWS credentials."""
        self.model_id = model_id or default_bedrock_model
        
        if self.model_id not in valid_bedrock_models:
            raise ValueError(
                f"Invalid model: {self.model_id}. Available Bedrock models: {', '.join(model for model in valid_bedrock_models)}"
            )

        self.system_prompt = system_prompt or default_system_message
        self.access_key_id = access_key_id or KEY_FILE_HANDLER.fetch_data(KeyValues.AWS_ACCESS_KEY_ID)
        self.secret_access_key = secret_access_key or KEY_FILE_HANDLER.fetch_data(KeyValues.AWS_SECRET_ACCESS_KEY)
        self.session_token = session_token or KEY_FILE_HANDLER.fetch_data(KeyValues.AWS_SESSION_TOKEN)
        self.region = region or KEY_FILE_HANDLER.fetch_data(KeyValues.AWS_REGION)

        if not (self.access_key_id and self.secret_access_key):
            try:
                boto3.setup_default_session(region_name=self.region)
            except (NoCredentialsError, PartialCredentialsError):
                raise ValueError("AWS credentials are not found. Please provide valid access keys or ensure your AWS credentials file is configured.")

        self.client = AnthropicBedrock(
            aws_access_key=self.access_key_id,
            aws_secret_key=self.secret_access_key,
            aws_session_token=self.session_token,
            aws_region=self.region
        )
        self.a_client = AsyncAnthropicBedrock(
            aws_access_key=self.access_key_id,
            aws_secret_key=self.secret_access_key,
            aws_session_token=self.session_token,
            aws_region=self.region
        )

    def load_model(self):
        """Loads the Bedrock client."""
        return self.client

    def extract_json(self, text: str) -> dict:
        """Attempts to parse the given text into a valid JSON dictionary."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Error decoding JSON")
            return {}

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict, None]:
        messages = [{"role": "user", "content": prompt}]
        full_prompt = self._build_prompt(schema)

        try:
            response = self.client.messages.create(
                model=self.model_id,
                messages=messages,
                system=full_prompt,
                max_tokens=1000,
            )

            generated_text = response.content[0].text if response.content else ""

            return self._parse_response(generated_text, schema)

        except Exception as e:
            logger.error(f"Error during sync generation: {e}")
            return {} if schema is None else None

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict, None]:
        messages = [{"role": "user", "content": prompt}]
        full_prompt = self._build_prompt(schema)

        try:
            response = await self.a_client.messages.create(
                model=self.model_id,
                messages=messages,
                system=full_prompt,
                max_tokens=1000,
            )

            generated_text = response.content[0].text if response.content else ""

            return self._parse_response(generated_text, schema)

        except Exception as e:
            logger.error(f"Error during async generation: {e}")
            return {} if schema is None else None

    def get_model_name(self):
        """Returns the model ID being used."""
        return self.model_id
    
    def _build_prompt(self, schema: Optional[BaseModel]) -> str:
        if schema:
            return f"{self.system_prompt}\nOutput JSON schema: {schema.model_json_schema()}"
        return self.system_prompt

    def _parse_response(self, generated_text: str, schema: Optional[BaseModel]) -> Union[BaseModel, dict, None]:
        if schema:
            try:
                extracted = self.extract_json(generated_text)
                return schema(**extracted)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                return None
        return generated_text
