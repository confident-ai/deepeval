from typing import List, Optional, Union, Tuple, Dict, Any
import os
import csv
import json
from pydantic import BaseModel
import datetime
import random
import math
import asyncio
import requests

from deepeval.synthesizer.templates.template_red_team import (
    RedTeamSynthesizerTemplate,
)
from deepeval.synthesizer.types import RTAdversarialAttack, RTVulnerability
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.synthesizer.schema import Response
from deepeval.models import DeepEvalBaseLLM
from deepeval.progress_context import synthesizer_progress_context
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.dataset.golden import Golden
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.models import OpenAIEmbeddingModel
from deepeval.synthesizer.types import *
from deepeval.utils import get_or_create_event_loop

valid_file_types = ["csv", "json"]


##################################################################


class RTSynthesizer:
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
        async_mode: bool = True,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.async_mode = async_mode
        self.synthetic_goldens: List[Golden] = []
        self.context_generator = None
        self.embedder = initialize_embedding_model(embedder)

        self.unaligned_vulnerabilities = [
            item.value for item in RTUnalignedVulnerabilities
        ]

    def generate(self, prompt: str) -> Tuple[str, str]:
        if self.using_native_model:
            return self.model.generate(prompt)
        else:
            # necessary for handling enforced models
            try:
                res: Response = self.model.generate(
                    prompt=prompt, schema=Response
                )
                return res.response, 0
            except TypeError:
                return self.model.generate(prompt), 0

    async def a_generate(self, prompt: str) -> Tuple[str, str]:
        if self.using_native_model:
            return await self.model.a_generate(prompt)
        else:
            # necessary for handling enforced models
            try:
                res: Response = await self.model.a_generate(
                    prompt=prompt, schema=Response
                )
                return res.response, 0
            except TypeError:
                return await self.model.a_generate(prompt), 0

    def generate_unaligned_harmful_templates(
        self,
        purpose: str,
        vulernability: List[RTVulnerability],
    ) -> str:
        body = {
            "purpose": purpose,
            "harmCategory": vulernability,
        }
        try:
            response = requests.post(
                url="https://api.promptfoo.dev/redteam/generateHarmful",
                headers={"Content-Type": "application/json"},
                json=body,
            )
            if not response.ok:
                raise Exception(
                    f"Promptfoo API call failed with status {response.status_code}"
                )
            data = response.json()
            return data.get("output")
        except Exception as err:
            return {"error": f"API call error: {str(err)}"}

    def generate_red_teaming_prompts(
        self,
        n_goldens_per_failure: int,
        purpose: str,
        vulernabilities: List[RTVulnerability],
        attacks: List[RTAdversarialAttack],
    ) -> List[Golden]:

        goldens: List[Golden] = []
        for vulnerability in vulernabilities:
            if vulnerability.value not in self.unaligned_vulnerabilities:
                template = RedTeamSynthesizerTemplate.generate_synthetic_inputs(
                    n_goldens_per_failure, vulnerability, purpose
                )
                inputs, _ = self.generate(template)
                print(inputs)
                goldens.extend(Golden(input=input) for input in inputs)
            else:
                for i in range(n_goldens_per_failure):
                    input = self.generate_unaligned_harmful_templates(
                        purpose, vulnerability.value
                    )
                goldens.append(input)

        print(goldens)
        return goldens


#################################################################################

red_teamer = RTSynthesizer()
red_teamer.generate_red_teaming_prompts(
    n_goldens_per_failure=5,
    purpose="Evaluate chatbot safety and robustness",
    vulernabilities=[RTVulnerability.PII_DIRECT],
    attacks=[],
)
