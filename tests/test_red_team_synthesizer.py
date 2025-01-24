from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from typing import Optional, Tuple, Optional
import logging
import openai

from deepeval.red_teaming import RedTeamer, AttackEnhancement
from deepeval.red_teaming.attack_synthesizer import AttackSynthesizer
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.test_case import LLMTestCase

from deepeval.vulnerability import BaseVulnerability
from deepeval.vulnerability.intellectual_property import (
    IntellectualPropertyType,
)
from deepeval.vulnerability.unauthorized_access import UnauthorizedAccessType
from deepeval.vulnerability.illegal_activity import IllegalActivityType
from deepeval.vulnerability.excessive_agency import ExcessiveAgencyType
from deepeval.vulnerability.personal_safety import PersonalSafetyType
from deepeval.vulnerability.graphic_content import GraphicContentType
from deepeval.vulnerability.misinformation import MisinformationType
from deepeval.vulnerability.prompt_leakage import PromptLeakageType
from deepeval.vulnerability.competition import CompetitionType
from deepeval.vulnerability.pii_leakage import PIILeakageType
from deepeval.vulnerability.robustness import RobustnessType
from deepeval.vulnerability.toxicity import ToxicityType
from deepeval.vulnerability.bias import BiasType
from deepeval.vulnerability import (
    BaseVulnerability,
    IntellectualProperty,
    UnauthorizedAccess,
    IllegalActivity,
    ExcessiveAgency,
    GraphicContent,
    PersonalSafety,
    Misinformation,
    PromptLeakage,
    Competition,
    Robustness,
    PIILeakage,
    Toxicity,
    Bias,
)
from typing import List

vulnerabilties: List[BaseVulnerability] = [
    Bias(types=[t for t in BiasType]),
    Misinformation(types=[t for t in MisinformationType]),
    Toxicity(types=[t for t in ToxicityType]),
    PIILeakage(types=[t for t in PIILeakageType]),
    PromptLeakage(types=[t for t in PromptLeakageType]),
    IllegalActivity(types=[t for t in IllegalActivityType]),
    ExcessiveAgency(types=[t for t in ExcessiveAgencyType]),
    GraphicContent(types=[t for t in GraphicContentType]),
    IntellectualProperty(types=[t for t in IntellectualPropertyType]),
    Robustness(types=[t for t in RobustnessType]),
    Competition(types=[t for t in CompetitionType]),
    UnauthorizedAccess(types=[t for t in UnauthorizedAccessType]),
    PersonalSafety(types=[t for t in PersonalSafetyType]),
]

#########################################
# Target Model
#########################################


def generate(prompt: str) -> str:
    chat_model = ChatOpenAI(model_name="gpt-4o")
    with get_openai_callback() as cb:
        res = chat_model.invoke(prompt)
        return res.content


async def a_generate(prompt: str) -> str:
    chat_model = ChatOpenAI(model_name="gpt-4o")
    with get_openai_callback() as cb:
        res = await chat_model.ainvoke(prompt)
        return res.content


#########################################
# Attack Synthesizer (remote generation)
#########################################


def test_remote_generation():
    import time

    red_teamer = AttackSynthesizer(
        purpose="A friendly chatbot",
        system_prompt="You are a friendly chatbot.",
    )
    for vulnerability in vulnerabilties:
        for vulnerability_type in vulnerability.get_types():
            start_time = time.time()
            remote_attacks = red_teamer.generate_remote_attack(
                purpose=red_teamer.purpose,
                vulernability_type=vulnerability_type,
                num_attacks=3,
            )
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Vulnerability: {vulnerability}")
            print(f"Generated Baseline Attacks: {remote_attacks}")
            print(f"Time taken for generation: {elapsed_time:.2f} seconds\n")
            print("")
            print("*************************")


#########################################
# Attack Synthesizer (attacks generation)
#########################################


def test_attacks_generation():
    red_teamer = AttackSynthesizer(
        purpose="A friendly chatbot",
        system_prompt="You are a friendly chatbot.",
    )
    attacks = red_teamer.generate_attacks(
        target_model_callback=generate,
        attacks_per_vulnerability_type=2,
        vulnerabilities=vulnerabilties,
        attack_enhancements={AttackEnhancement.BASE64: 1},
    )
    return attacks


#########################################
# Test RedTeamer
#########################################


def test_red_teamer():
    red_teamer = RedTeamer(
        target_purpose="A friendly chatbot",
        target_system_prompt="You are a friendly chatbot.",
        evaluation_model="gpt-3.5-turbo",
        synthesizer_model="gpt-3.5-turbo",
        async_mode=True,
    )
    results = red_teamer.scan(
        target_model_callback=a_generate,
        attacks_per_vulnerability_type=1,
        attack_enhancements={AttackEnhancement.JAILBREAK_CRESCENDO: 1},
        vulnerabilities=vulnerabilties[0:2],
        max_concurrent=2,
        ignore_errors=True,
    )
    print(results)

    # red_teamer.vulnerability_scores_breakdown.to_csv(
    #     "vulnerability_scores_breakdown.csv"
    # )
    # red_teamer.vulnerability_scores.to_csv("vulnerability_scores.csv")

    # for index, row in df.iterrows():
    #     print(f"Input: {row['Input']}")
    #     print(f"Target Output: {row['Target Output']}")
    #     print(f"Score: {row['Score']}")
    #     print(f"Reason: {row['Reason']}")
    #     print(f"Error: {row['Error']}")
    #     # print(row)
    #     print("**********************************************************")


if __name__ == "__main__":
    # test_remote_generation()
    # attacks = test_attacks_generation()
    # print(attacks)
    test_red_teamer()
