"""
Enhanced GEval with oversampling support for token probability estimation.

This module provides an enhanced GEval class that supports oversampling for
token probability estimation when custom models don't support logprobs.
"""

import math
import re
from typing import Optional, Tuple, Union, Dict, List

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval.g_eval import GEval

# Optional pydantic import
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object


class GEvalWithOversampling(GEval):
    """
    Enhanced GEval that supports oversampling for token probability estimation
    when custom models don't support logprobs.
    
    This implements the MLE (Maximum Likelihood Estimation) approach mentioned
    in the GEval paper for models that can't provide token probabilities directly.
    """
    
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        rubric: Optional[List] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
        # New parameters for oversampling
        oversampling_enabled: bool = False,
        oversampling_count: int = 10,
    ):
        super().__init__(
            name=name,
            evaluation_params=evaluation_params,
            criteria=criteria,
            evaluation_steps=evaluation_steps,
            rubric=rubric,
            model=model,
            threshold=threshold,
            top_logprobs=top_logprobs,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            _include_g_eval_suffix=_include_g_eval_suffix,
        )
        self.oversampling_enabled = oversampling_enabled
        self.oversampling_count = oversampling_count
    
    def _estimate_token_probabilities_with_oversampling(
        self, 
        prompt: str, 
        target_score: int
    ) -> Dict[int, float]:
        """
        Estimate token probabilities using oversampling when logprobs aren't available.
        
        This implements the MLE approach mentioned in the GEval paper.
        It generates multiple responses and calculates empirical probabilities.
        
        Args:
            prompt: The evaluation prompt
            target_score: The target score to estimate probabilities for
            
        Returns:
            Dictionary mapping scores to their estimated probabilities
        """
        if not self.oversampling_enabled:
            return {target_score: 1.0}  # No oversampling, assume deterministic
        
        # Generate multiple responses to estimate probabilities
        responses = []
        for _ in range(self.oversampling_count):
            try:
                response, _ = self.model.generate(prompt)
                # Parse the response to extract the score
                score = self._extract_score_from_response(response)
                responses.append(score)
            except Exception as e:
                if self.verbose_mode:
                    print(f"Error in oversampling iteration: {e}")
                continue
        
        # Calculate empirical probabilities
        score_counts = {}
        total_responses = len(responses)
        
        if total_responses == 0:
            # If no successful responses, return deterministic
            return {target_score: 1.0}
        
        for score in responses:
            score_counts[score] = score_counts.get(score, 0) + 1
        
        # Convert counts to probabilities
        probabilities = {}
        for score, count in score_counts.items():
            probabilities[score] = count / total_responses
        
        return probabilities
    
    def _extract_score_from_response(self, response: str) -> int:
        """
        Extract numeric score from model response.
        
        This is a simple implementation that looks for the first number in the response.
        Custom models may need to override this method for more sophisticated parsing.
        
        Args:
            response: The model's response
            
        Returns:
            Extracted score as integer
        """
        # Look for numbers in the response
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        
        # If no numbers found, try to extract from common patterns
        # This is a basic implementation - custom models may need more sophisticated parsing
        
        # Look for score patterns like "Score: 5" or "Rating: 4"
        score_patterns = [
            r'score[:\s]*(\d+)',
            r'rating[:\s]*(\d+)',
            r'grade[:\s]*(\d+)',
            r'(\d+)/10',
            r'(\d+)/5',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                return int(match.group(1))
        
        # Default fallback
        return 5
    
    def _calculate_weighted_score_with_oversampling(
        self, 
        prompt: str, 
        raw_score: int
    ) -> float:
        """
        Calculate weighted score using oversampling when logprobs aren't available.
        
        This method estimates token probabilities through multiple generations
        and calculates a weighted average score.
        
        Args:
            prompt: The evaluation prompt
            raw_score: The raw score from the model
            
        Returns:
            Weighted score based on estimated probabilities
        """
        probabilities = self._estimate_token_probabilities_with_oversampling(prompt, raw_score)
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_probability = 0.0
        
        for score, probability in probabilities.items():
            weighted_sum += score * probability
            total_probability += probability
        
        if total_probability > 0:
            return weighted_sum / total_probability
        else:
            return raw_score  # Fallback to raw score
    
    def _evaluate(
        self, test_case: LLMTestCase, _additional_context: Optional[str] = None
    ) -> Tuple[Union[int, float], str]:
        """
        Enhanced evaluation that supports oversampling for token probability estimation.
        
        This method extends the base GEval evaluation to include oversampling
        when logprobs aren't available from the model.
        """
        # Get the prompt using the parent class method
        prompt = self._get_evaluation_prompt(test_case, _additional_context)
        
        try:
            # Try logprobs first (existing behavior)
            if hasattr(self.model, 'generate_raw_response'):
                res, cost = self.model.generate_raw_response(
                    prompt, top_logprobs=self.top_logprobs
                )
                self.evaluation_cost += cost
                data = self._trim_and_load_json(res.choices[0].message.content)
                
                reason = data["reason"]
                score = data["score"]
                
                if self.strict_mode:
                    return score, reason
                
                # Use existing weighted scoring logic
                try:
                    from deepeval.metrics.g_eval.utils import calculate_weighted_summed_score
                    weighted_summed_score = calculate_weighted_summed_score(score, res)
                    return weighted_summed_score, reason
                except:
                    return score, reason
            
            # Fallback to oversampling if enabled
            elif self.oversampling_enabled:
                # Use basic generation
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = self._trim_and_load_json(res)
                
                reason = data["reason"]
                score = data["score"]
                
                if self.strict_mode:
                    return score, reason
                
                # Use oversampling for weighted scoring
                weighted_score = self._calculate_weighted_score_with_oversampling(prompt, score)
                return weighted_score, reason
            
            else:
                # Fallback to basic evaluation (existing behavior)
                return super()._evaluate(test_case, _additional_context)
                
        except Exception as e:
            if self.verbose_mode:
                print(f"Error in enhanced evaluation: {e}")
            # Fallback to basic evaluation
            return super()._evaluate(test_case, _additional_context)
    
    def _trim_and_load_json(self, response: str) -> Dict:
        """Helper method to trim and load JSON from response"""
        import json
        
        # Clean the response
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        return json.loads(cleaned_response.strip()) 