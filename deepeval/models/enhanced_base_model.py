"""
Enhanced base model with generate_raw_response() support for token probability estimation.

This module provides an enhanced base class that supports token probability estimation
for custom models in DeepEval.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union, Dict
try:
    from openai.types.chat import ChatCompletion
except ImportError:
    # Fallback for when openai package is not available
    ChatCompletion = Any

from deepeval.models.base_model import DeepEvalBaseLLM


class EnhancedDeepEvalBaseLLM(DeepEvalBaseLLM):
    """
    Enhanced base class that provides generate_raw_response() method.
    
    Custom models can override this to provide token probability support for GEval.
    This enables the weighted scoring methodology described in the GEval paper.
    """
    
    def load_model(self, *args, **kwargs):
        """Default implementation - override in subclasses"""
        return None
    
    def generate(self, *args, **kwargs) -> str:
        """Default implementation - override in subclasses"""
        raise NotImplementedError("Subclasses must implement generate()")
    
    async def a_generate(self, *args, **kwargs) -> str:
        """Default implementation - override in subclasses"""
        raise NotImplementedError("Subclasses must implement a_generate()")
    
    def get_model_name(self, *args, **kwargs) -> str:
        """Default implementation - override in subclasses"""
        return self.model_name
    
    def generate_raw_response(
        self, 
        prompt: str, 
        top_logprobs: int = 20
    ) -> Tuple[ChatCompletion, float]:
        """
        Generate response with token probabilities for GEval.
        
        This method should:
        1. Call your model's API with logprobs=True
        2. Return a ChatCompletion object with logprobs information
        3. Include cost calculation
        
        Args:
            prompt: The input prompt
            top_logprobs: Number of top logprobs to return
            
        Returns:
            Tuple of (ChatCompletion, cost)
            
        Raises:
            NotImplementedError: If the model doesn't support logprobs
        """
        raise NotImplementedError(
            "Custom models should implement generate_raw_response() "
            "for token probability support, or use GEvalWithOversampling"
        )
    
    async def a_generate_raw_response(
        self, 
        prompt: str, 
        top_logprobs: int = 20
    ) -> Tuple[ChatCompletion, float]:
        """
        Async version of generate_raw_response.
        
        Args:
            prompt: The input prompt
            top_logprobs: Number of top logprobs to return
            
        Returns:
            Tuple of (ChatCompletion, cost)
        """
        raise NotImplementedError(
            "Custom models should implement a_generate_raw_response() "
            "for async token probability support"
        )


def create_chat_completion_with_logprobs(
    content: str,
    token_logprobs: Dict[str, Dict[str, float]],
    prompt_tokens: int = 10,
    completion_tokens: int = 5
) -> ChatCompletion:
    """
    Helper function to create a ChatCompletion object with logprobs.
    
    This is useful for custom models that need to create ChatCompletion-like objects
    with token probability information.
    
    Args:
        content: The response content
        token_logprobs: Dictionary mapping tokens to their top logprobs
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        ChatCompletion object with logprobs
    """
    # This is a simplified implementation
    # In practice, you'd create the actual ChatCompletion structure
    # based on your model's API response
    
    class MockChatCompletion:
        def __init__(self, content: str, logprobs: Dict[str, Any]):
            self.choices = [MockChoice(content, logprobs)]
            self.usage = MockUsage(prompt_tokens, completion_tokens)

    class MockChoice:
        def __init__(self, content: str, logprobs: Dict[str, Any]):
            self.message = MockMessage(content)
            self.logprobs = MockLogprobs(logprobs)

    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockLogprobs:
        def __init__(self, logprobs: Dict[str, Any]):
            self.content = [MockTokenLogprob(token, probs) for token, probs in logprobs.items()]

    class MockTokenLogprob:
        def __init__(self, token: str, top_logprobs: Dict[str, float]):
            self.token = token
            self.top_logprobs = [MockTopLogprob(t, p) for t, p in top_logprobs.items()]

    class MockTopLogprob:
        def __init__(self, token: str, logprob: float):
            self.token = token
            self.logprob = logprob

    class MockUsage:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
    
    return MockChatCompletion(content, token_logprobs) 