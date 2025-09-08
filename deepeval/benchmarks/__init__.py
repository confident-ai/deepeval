from .big_bench_hard.big_bench_hard import BigBenchHard
from .mmlu.mmlu import MMLU
from .hellaswag.hellaswag import HellaSwag
from .drop.drop import DROP
from .truthful_qa.truthful_qa import TruthfulQA
from .human_eval.human_eval import HumanEval
from .squad.squad import SQuAD
from .gsm8k.gsm8k import GSM8K
from .math_qa.math_qa import MathQA
from .logi_qa.logi_qa import LogiQA
from .bool_q.bool_q import BoolQ
from .arc.arc import ARC
from .bbq.bbq import BBQ
from .lambada.lambada import LAMBADA
from .winogrande.winogrande import Winogrande
from .equity_med_qa.equity_med_qa import EquityMedQA
from .ifeval.ifeval import IFEval
from .turing_test.turing_test import TuringTest
from .ai_belief_test.ai_belief_test import AIBeliefTest

__all__ = [
    "BigBenchHard",
    "MMLU",
    "HellaSwag",
    "DROP",
    "TruthfulQA",
    "HumanEval",
    "SQuAD",
    "GSM8K",
    "MathQA",
    "LogiQA",
    "BoolQ",
    "ARC",
    "BBQ",
    "LAMBADA",
    "Winogrande",
    "EquityMedQA",
    "IFEval",
    "TuringTest",
    "AIBeliefTest",
]
