import random
from typing import List, Optional, Union, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model
)
from deepeval.metrics.adversarial_robustness.schema import AdversarialRobustnessScoreReason, PerturbationResult
from deepeval.metrics.adversarial_robustness.template import AdversarialRobustnessTemplate

# Optional dependency handling
try:
    import gensim.downloader as api
    import numpy as np
    from nltk.tokenize import word_tokenize
    import nltk
except ImportError:
    gensim_available = False
else:
    gensim_available = True

class AdversarialRobustnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(
        self,
        threshold: float = 0.5,
        model_to_test: Union[str, DeepEvalBaseLLM] = None,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        perturbation_type: str = 'semantic',
        n_perturbations: int = 10,
    ):
        if not model_to_test:
            raise ValueError("`model_to_test` must be provided to test for adversarial robustness.")
            
        self.threshold = 1 if strict_mode else threshold
        
        # This is the model we are testing FOR robustness
        self.model_to_test, _ = initialize_model(model_to_test)
        
        # This is the model used to generate the final reason
        self.evaluation_model, self.using_native_model = initialize_model(evaluation_model)
        
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        
        if perturbation_type not in ['semantic', 'orthographic']:
            raise ValueError("`perturbation_type` must be either 'semantic' or 'orthographic'.")
        self.perturbation_type = perturbation_type
        
        if self.perturbation_type == 'semantic' and not gensim_available:
            raise ImportError("Semantic perturbations require `gensim`, `numpy`, and `nltk`. Please install them with: pip install 'deepeval[adversarial]'")

        self.n_perturbations = n_perturbations
        self.word2vec_model = None

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator, _in_component=_in_component):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(self.a_measure(test_case, _show_indicator=False, _in_component=_in_component))
            else:
                self._run_sync_measure(test_case)
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, async_mode=True, _show_indicator=_show_indicator, _in_component=_in_component):
            await self._run_async_measure(test_case)
        return self.score

    def _run_sync_measure(self, test_case: LLMTestCase):
        perturbed_inputs = self._generate_perturbations(test_case.input)
        
        results = []
        for p_input in perturbed_inputs:
            perturbed_output = self.model_to_test.generate(p_input)[0]
            is_robust = (perturbed_output.strip().lower() == test_case.actual_output.strip().lower())
            results.append(PerturbationResult(
                original_output=test_case.actual_output,
                perturbed_input=p_input,
                perturbed_output=perturbed_output,
                is_robust=is_robust
            ))
            
        self.score = self._calculate_score(results)
        if self.include_reason:
            self.reason = self._generate_reason(results)
            
        self.success = self.score >= self.threshold
        self.verbose_logs = construct_verbose_logs(self, steps=[
            f"Perturbation Type: {self.perturbation_type}",
            f"Perturbation Results:\n{prettify_list(results)}",
            f"Score: {self.score}\nReason: {self.reason}",
        ])

    async def _run_async_measure(self, test_case: LLMTestCase):
        perturbed_inputs = self._generate_perturbations(test_case.input)
        
        predictions = await self.model_to_test.a_generate(perturbed_inputs)
        
        results = []
        for i, p_input in enumerate(perturbed_inputs):
            perturbed_output = predictions[i]
            is_robust = (perturbed_output.strip().lower() == test_case.actual_output.strip().lower())
            results.append(PerturbationResult(
                original_output=test_case.actual_output,
                perturbed_input=p_input,
                perturbed_output=perturbed_output,
                is_robust=is_robust
            ))
            
        self.score = self._calculate_score(results)
        if self.include_reason:
            self.reason = await self._a_generate_reason(results)

        self.success = self.score >= self.threshold
        self.verbose_logs = construct_verbose_logs(self, steps=[
            f"Perturbation Type: {self.perturbation_type}",
            f"Perturbation Results:\n{prettify_list(results)}",
            f"Score: {self.score}\nReason: {self.reason}",
        ])

    def _calculate_score(self, results: List[PerturbationResult]) -> float:
        if not results:
            return 1.0
        robust_count = sum(1 for r in results if r.is_robust)
        score = robust_count / len(results)
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_reason(self, results: List[PerturbationResult]) -> str:
        if not results:
            return "No perturbations were generated, so robustness is considered perfect."
            
        prompt = AdversarialRobustnessTemplate.generate_reason(
            score=self.score,
            perturbation_type=self.perturbation_type,
            results=results
        )
        res, cost = self.evaluation_model.generate(prompt, schema=AdversarialRobustnessScoreReason)
        self.evaluation_cost += cost
        return res.reason

    async def _a_generate_reason(self, results: List[PerturbationResult]) -> str:
        if not results:
            return "No perturbations were generated, so robustness is considered perfect."
            
        prompt = AdversarialRobustnessTemplate.generate_reason(
            score=self.score,
            perturbation_type=self.perturbation_type,
            results=results
        )
        res, cost = await self.evaluation_model.a_generate(prompt, schema=AdversarialRobustnessScoreReason)
        self.evaluation_cost += cost
        return res.reason

    def _generate_perturbations(self, text: str) -> List[str]:
        if self.perturbation_type == 'semantic':
            return self._generate_semantic_perturbations(text)
        else: # orthographic
            return self._generate_orthographic_perturbations(text)

    def _load_word2vec_model(self):
        if self.word2vec_model is None:
            print("Downloading Word2Vec model ('word2vec-google-news-300')...")
            self.word2vec_model = api.load('word2vec-google-news-300')
            print("Model downloaded.")
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

    def _generate_semantic_perturbations(self, text: str) -> List[str]:
        self._load_word2vec_model()
        tokens = word_tokenize(text)
        perturbed_texts = set()

        for _ in range(self.n_perturbations):
            new_tokens = list(tokens)
            # Select a random word to perturb
            word_idx = random.choice(range(len(new_tokens)))
            word_to_replace = new_tokens[word_idx]

            if word_to_replace in self.word2vec_model:
                # Find similar words (synonyms)
                similar_words = self.word2vec_model.most_similar(word_to_replace, topn=5)
                if similar_words:
                    # Choose a random synonym
                    synonym = random.choice(similar_words)[0]
                    new_tokens[word_idx] = synonym
                    perturbed_texts.add(" ".join(new_tokens))

        return list(perturbed_texts)

    def _generate_orthographic_perturbations(self, text: str) -> List[str]:
        perturbed_texts = set()
        for _ in range(self.n_perturbations):
            chars = list(text)
            if not chars: continue
            
            idx = random.randint(0, len(chars) - 1)
            char_to_perturb = chars[idx]
            
            # Simple perturbation: swap with a random alphanumeric character
            if char_to_perturb.isalpha():
                new_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                chars[idx] = new_char.upper() if char_to_perturb.isupper() else new_char
            elif char_to_perturb.isdigit():
                chars[idx] = random.choice('0123456789')
            
            perturbed_texts.add("".join(chars))
            
        return list(perturbed_texts)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            self.success = self.score >= self.threshold
        return self.success
        
    @property
    def __name__(self):
        return "Adversarial Robustness"