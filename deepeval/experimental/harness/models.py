# The idea here is to have a middleground level of abstraction so that users can use any 
# type of model (vllm, llamacpp, huggingface etc, without doing much hassel). All they 
# need to do is paste the inference code and tokenizer and they can get their evaluation done. 

# todo: This version does not supports batching currently. This will be done in next versions 

import sys
import json
import tiktoken
from pathlib import Path 
from typing import Optional, Union, List 

from lm_eval import base, evaluator
import lm_eval.tasks as lm_eval_tasks
from lm_eval.base import BaseLM

from transformers import AutoTokenizer
from deepeval.experimental.harness.config import GeneralConfig, APIEndpointConfig

class DeepEvalBaseHarnessWrapper(BaseLM):
    def __init__(self, config: Union[GeneralConfig, APIEndpointConfig]) -> None:
        """A custom lm-eval harness wrapper which supports deepeval 

        Args:
            config (Union[GeneralConfig, APIEndpointConfig]): General Config is for the common LLM generation configs 
                and APIEndpointConfig expects to put the Open AI authorization configs or any server 
                that is proxied by Open AI 
        """
        
        assert isinstance(config, APIEndpointConfig) or isinstance(config, GeneralConfig), ValueError(
            'Config can be either be from GeneralConfig or APIEndpointConfig'
        )
        
        self.config = config
        self._model_type = 'openai' if isinstance(config, APIEndpointConfig) else 'hf'
        
        # init model and tokenizer
        self.tokenizer = self._init_tokenizer()
        self.model = self.init_model()
        
    def init_model(self):
        """Function that is to be used while inheriting this base class. 
        
        Please override this function such that you initialize self.model
        which is an object which can be from huggingface/openai/vllm etc. 
        """
        return self  
    
    def generate(self):
        """Function that is to be used while inheriting this base class. 
        
        Please override this function such that you can have your own custom generate function
        with the given config or even with more custom kwargs 
        """
        raise NotImplementedError('Needs to be implemented from user side')
    
    def _init_tokenizer(self):
        """Initializing tokenizer. This tokenizer can be either from tiktoken or a huggingface tokenizer 
        or from a path (downloaded from huggingface models)
        """
        if self.config.tokenizer_name_or_path == 'tiktoken' and self._model_type == 'openai':
            self.tokenizer = tiktoken.encoding_for_model(self.config.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name_or_path)
        return self 
    
    @property
    def eot_token_id(self):
        # need to figure out what is the default eos for tiktoken 
        return 1 if self._model_type == 'openai' else self.tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return self.config.context_length

    @property
    def vocab_size(self):
        # todo: need to figure out the vocab size of Open AI or an estimated one 
        return self.tokenizer.vocab_size
    
    @property
    def max_gen_toks(self):
        return self.config.max_generation_length

    @property
    def batch_size(self):
        # todo: Right now there is no batching support for GPU. Kept for next versions
        return 1
    
    @property
    def device(self):
        if self._model_type == 'openai':
            return 'cpu'
        else:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def tok_encode(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt)
    
    @property
    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def run_eval(
        self, eval_tasks: List[str], bootstrap_iters: int, 
        num_fewshot: Optional[int] = None, 
        limit: Optional[int] = None, no_cache: bool = True 
    ):
        # todo: Caching is not available in the current version

        import fnmatch
        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        eval_tasks = pattern_match(eval_tasks, lm_eval_tasks.ALL_TASKS)
        lm_eval_tasks.get_task_dict(eval_tasks)
        
        lm = self 
        
        results = evaluator.evaluate(
            lm=lm,
            task_dict=eval_tasks.get_task_dict(eval_tasks),
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters
        )
        
        results["config"] = dict(
            model=self.config.model if self._model_type == 'openai' else self.config.model_name,
            batch_size = self.batch_size,
            device=str(self.device),
            num_fewshot=self.config.n_samples if num_fewshot is None else num_fewshot,
            limit=self.config.limit if limit is None else limit,
            bootstrap_iters=bootstrap_iters,
            no_cache=no_cache
        )
        return results 
    
    def deep_evaluate():
        pass 