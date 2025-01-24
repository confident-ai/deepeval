from typing import Optional, List, Dict, Tuple, Union
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Image, HarmCategory, HarmBlockThreshold
from pydantic_openapi_schema.v3.v3_1_0 import Schema

from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

valid_gemini_models = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.0-pro",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.0-pro-vision",
    "gemini-1.0-pro-vision-001"
]

valid_multimodal_gemini_models = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.0-pro-vision",
    "gemini-1.0-pro-vision-001"
]

default_gemini_model = "gemini-1.5-pro"
default_multimodal_gemini_model = "gemini-1.5-pro"

class GeminiModel(DeepEvalBaseLLM):
    """Class that implements Google Vertex AI Gemini models for text-based evaluation.
    
    This class provides integration with Google's Gemini models through Vertex AI,
    supporting text-only inputs for evaluation tasks.
    
    Attributes:
        model_name: Name of the Gemini model to use
        project_id: Google Cloud project ID
        location: Google Cloud region
        
    Example:
        ```python
        from deepeval.models import GeminiModel
        
        # Initialize the model
        model = GeminiModel(
            model_name="gemini-1.5-pro-001",
            project_id="your-project-id",
            location="us-central1"
        )
        
        # Generate text
        response = model.generate("What is the capital of France?")
        ```
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        *args,
        **kwargs
    ):
        model_name = model_name or default_gemini_model
        if model_name not in valid_gemini_models:
            raise ValueError(
                f"Invalid model. Available Gemini models: {', '.join(model for model in valid_gemini_models)}"
            )
            
        # Get credentials from key handler if not provided
        self.project_id = project_id or KEY_FILE_HANDLER.fetch_data(KeyValues.GOOGLE_CLOUD_PROJECT)
        self.location = location or KEY_FILE_HANDLER.fetch_data(KeyValues.GOOGLE_CLOUD_LOCATION)
        
        if not self.project_id or not self.location:
            raise ValueError(
                "Google Cloud project_id and location are required. Either provide them directly "
                "or set them in your DeepEval configuration."
            )
            
        # Initialize Vertex AI with project and location
        vertexai.init(project=self.project_id, location=self.location)
            
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Loads and initializes the Gemini model.
        
        Returns:
            A GenerativeModel instance configured with safety settings optimized for evaluation.
        """
        # Initialize safety filters - set to minimum for evaluation purposes
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }

        return GenerativeModel(
            model_name=self.model_name,
            safety_settings=safety_settings
        )

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None
    ) -> str:
        """Generates text from a prompt.
        
        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output
            
        Returns:
            Generated text response or structured output as Pydantic model
        """
        if schema is not None:
            # Convert Pydantic model to OpenAPI schema
            schema_dict = Schema.from_pydantic(schema).model_dump()
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict
                )
            )
            
            # Parse response back into Pydantic model
            return schema.model_validate_json(response.text)
        else:
            response = self.model.generate_content(prompt)
            return response.text

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None
    ) -> str:
        """Asynchronously generates text from a prompt.
        
        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output
            
        Returns:
            Generated text response or structured output as Pydantic model
        """
        if schema is not None:
            # Convert Pydantic model to OpenAPI schema
            schema_dict = Schema.from_pydantic(schema).model_dump()
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict
                )
            )
            
            # Parse response back into Pydantic model
            return schema.model_validate_json(response.text)
        else:
            response = await self.model.generate_content_async(prompt)
            return response.text

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name


class MultimodalGeminiModel(DeepEvalBaseMLLM):
    """Class that implements Google Vertex AI Gemini models for multimodal evaluation.
    
    This class provides integration with Google's Gemini models through Vertex AI,
    supporting both text and multimodal (text + image) inputs for evaluation tasks.
    
    Attributes:
        model_name: Name of the Gemini model to use
        project_id: Google Cloud project ID
        location: Google Cloud region
        
    Example:
        ```python
        from deepeval.models import MultimodalGeminiModel
        
        # Initialize the model
        model = MultimodalGeminiModel(
            model_name="gemini-pro-vision",
            project_id="your-project-id",
            location="us-central1"
        )
        
        # Generate text from text + image input
        response = model.generate([
            "Describe what you see in this image:",
            MLLMImage(url="path/to/image.jpg", local=True)
        ])
        ```
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        *args,
        **kwargs
    ):
        model_name = model_name or default_multimodal_gemini_model
        if model_name not in valid_multimodal_gemini_models:
            raise ValueError(
                f"Invalid model. Available Multimodal Gemini models: {', '.join(model for model in valid_multimodal_gemini_models)}"
            )
            
        # Get credentials from key handler if not provided
        self.project_id = project_id or KEY_FILE_HANDLER.fetch_data(KeyValues.GOOGLE_CLOUD_PROJECT)
        self.location = location or KEY_FILE_HANDLER.fetch_data(KeyValues.GOOGLE_CLOUD_LOCATION)
        
        if not self.project_id or not self.location:
            raise ValueError(
                "Google Cloud project_id and location are required. Either provide them directly "
                "or set them in your DeepEval configuration."
            )
            
        # Initialize Vertex AI with project and location
        vertexai.init(project=self.project_id, location=self.location)
            
        super().__init__(model_name, *args, **kwargs)
        self.model = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Loads and initializes the Gemini model.
        
        Returns:
            A GenerativeModel instance configured with safety settings optimized for evaluation.
        """
        # Initialize safety filters - set to minimum for evaluation purposes
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }

        return GenerativeModel(
            model_name=self.model_name,
            safety_settings=safety_settings
        )

    def generate_prompt(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ) -> List[Union[str, Part]]:
        """Converts DeepEval multimodal input into Vertex AI compatible format.
        
        Args:
            multimodal_input: List of strings and MLLMImage objects
            
        Returns:
            List of strings and Vertex AI Part objects ready for model input
            
        Raises:
            ValueError: If an invalid input type is provided
        """
        prompt = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append(ele)
            elif isinstance(ele, MLLMImage):
                if ele.local:
                    image = Part.from_image(Image.load_from_file(ele.url))
                else:
                    image = Part.from_uri(uri=ele.url, mime_type="image/jpeg")
                prompt.append(image)
            else:
                raise ValueError(f"Invalid input type: {type(ele)}")

        return prompt

    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None
    ) -> str:
        """Generates text from multimodal input.
        
        Args:
            multimodal_input: List of strings and MLLMImage objects
            schema: Optional Pydantic model for structured output
            
        Returns:
            Generated text response
        """
        prompt = self.generate_prompt(multimodal_input)
        
        if schema is not None:
            # Convert Pydantic model to OpenAPI schema
            schema_dict = Schema.from_pydantic(schema).model_dump()
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict
                )
            ) 
        else:
            response = self.model.generate_content(prompt)
        
        return response.text

    async def a_generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None
    ) -> str:
        """Asynchronously generates text from multimodal input.
        
        Args:
            multimodal_input: List of strings and MLLMImage objects
            schema: Optional Pydantic model for structured output
            
        Returns:
            Generated text response
        """
        prompt = self.generate_prompt(multimodal_input)
        
        if schema is not None:
            # Convert Pydantic model to OpenAPI schema
            schema_dict = Schema.from_pydantic(schema).model_dump()
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict
                )
            )         
        else:
            response = await self.model.generate_content_async(prompt)
        
        return response.text

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name
