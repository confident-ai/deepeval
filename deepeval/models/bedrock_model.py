import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import json
from pydantic import BaseModel, ValidationError
import logging
from typing import List, Tuple, Union, Optional
import base64
from io import BytesIO
import mimetypes
import requests
from PIL import Image as PILImage

from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseMLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.test_case import MLLMImage

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

valid_bedrock_models = [
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
default_system_message = "You are a helpful AI assistant. Always generate your response as a valid json. No explanation is needed just the json."

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
        
        if model_id not in valid_bedrock_models:
            raise ValueError(
                f"Invalid model: {model_id}. Available Bedrock models: {', '.join(model for model in valid_bedrock_models)}"
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

        self.client = boto3.client(
            "bedrock-runtime", 
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            aws_session_token=self.session_token if self.session_token else None
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

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict]:
        """Generates text using the Bedrock model and returns the response as a Pydantic model."""
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": messages,
            "system": self.system_prompt
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))

            content = response_body.get("content", [])
            if content and isinstance(content, list):
                generated_text = content[0].get('text', '')
            else:
                logger.error("Invalid response structure: 'content' not found or malformed")
                generated_text = ""
            
            if schema:
                try:
                    extracted_result = self.extract_json(generated_text)
                    return schema(**extracted_result)
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
                    return None
            return generated_text
        
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            return {} if schema is None else None
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict]:
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Returns the model ID being used."""
        return self.model_id
    

class MultimodalBedrockModel(DeepEvalBaseMLLM):
    """
    A class to interact with AWS Bedrock models for multimodal (text + image) evaluation.

    This class integrates AWS Bedrock, allowing both text and image inputs for generating multimodal outputs. It supports 
    both local and remote image inputs, converting images to base64 encoding for Bedrock requests.

    Attributes:
        model_id (str): The ID of the Bedrock model to use.
        access_key_id (str): AWS Access Key ID.
        secret_access_key (str): AWS Secret Access Key.
        session_token (str): AWS Session Token.
        region (str): The AWS region for the Bedrock service.
        client (boto3.client): The Bedrock client instance used for model interaction.
    
    Example:
        ```python
        from deepeval.models import MultimodalBedrockModel
        
        # Initialize the model
        model = MultimodalBedrockModel(
            model_id="your-bedrock-model-id",
            access_key_id="your-aws-access-key",
            secret_access_key="your-aws-secret-key",
            region="us-west-2"
        )
        
        # Generate a response based on text and image input
        response = model.generate([
            "Describe what you see in this image:",
            MLLMImage(url="path/to/image.jpg", local=True)
        ])
        ```

    Methods:
        __init__: Initializes the model, setting up credentials and the Bedrock client.
        load_model: Loads and returns the Bedrock client instance.
        encode_pil_image: Encodes a PIL image to base64 string format (JPEG).
        generate_prompt: Constructs a request payload from text and image inputs.
        generate: Sends a synchronous request to the Bedrock API for text and image-based generation.
        a_generate: Asynchronous wrapper for the `generate` method.
        get_model_name: Returns the Bedrock model ID in use.
    """
    def __init__(
        self,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
        *args,
        **kwargs
    ):
        self.model_id = model_id or default_multimodal_bedrock_model
        if model_id not in valid_bedrock_models:
            raise ValueError(
                f"Invalid model. Available Bedrock models: {', '.join(model for model in valid_bedrock_models)}"
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

        self.client = boto3.client(
            "bedrock-runtime", 
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            aws_session_token=self.session_token if self.session_token else None
        )

        super().__init__(model_id, *args, **kwargs)
        self.model = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Loads and initializes the Bedrock model.
        
        Returns:
            A Bedrock model instance ready for evaluation.
        """
        return self.client

    def encode_pil_image(self, pil_image: PILImage) -> str:
        """Convert a PIL image to a base64-encoded string (JPEG format)."""
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

    def generate_prompt(self, multimodal_input: List[Union[str, MLLMImage]], schema: Optional[BaseModel] = None) -> List[dict]:
        """Constructs the message payload with both text and image inputs."""
        prompt = []
        for item in multimodal_input:
            if isinstance(item, str):
                prompt.append({"type": "text", "text": item})
            elif isinstance(item, MLLMImage):
                if item.local:
                    image = PILImage.open(item.url)
                    image_data = self.encode_pil_image(image)
                    prompt.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                else:
                    prompt.append({"type": "image_url", "image_url": {"url": item.url}})
        return prompt

    def generate(self, multimodal_input: List[Union[str, MLLMImage]], schema: Optional[BaseModel] = None) -> Tuple[str, Optional[float]]:
        """Sends a synchronous request to Bedrock for text & image-based generation."""
        
        content_list = []
        
        for item in multimodal_input:
            if isinstance(item, str):

                content_list.append({"type": "text", "text": item})
            elif isinstance(item, MLLMImage):
                try:
                    media_type = mimetypes.guess_type(item.url)[0] or "image/jpeg"

                    if item.local:
                        with open(item.url, "rb") as img_file:
                            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    else:
                        response = requests.get(item.url)
                        image_base64 = base64.b64encode(response.content).decode("utf-8")
                    
                    content_list.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    })
                    logger.info(f"Processed image: {item.url} with media type {media_type}")
                except Exception as img_error:
                    logger.error(f"Error processing image {item.url}: {img_error}")
                    raise

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": content_list}],
            "system": self.system_prompt
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))

            content = response_body.get("content", [])
            if content and isinstance(content, list):
                generated_text = content[0].get('text', '')
            else:
                logger.error("Invalid response structure: 'content' not found or malformed")
                generated_text = ""

            if schema:
                try:
                    return schema(**generated_text)
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
                    return None
            return generated_text

        except Exception as e:
            logger.error(f"Error during Bedrock model inference: {e}")
            raise

    async def a_generate(self, multimodal_input: List[Union[str, MLLMImage]]) -> Tuple[str, Optional[float]]:
        """Async wrapper for the generate function."""
        return self.generate(multimodal_input)

    def get_model_name(self) -> str:
        return self.model_id
