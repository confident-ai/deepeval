from .big_bench_hard.big_bench_hard import BigBenchHard as BigBenchHard
from .mmlu.mmlu import MMLU as MMLU
from .hellaswag.hellaswag import HellaSwag as HellaSwag
from .drop.drop import DROP as DROP
from .truthful_qa.truthful_qa import TruthfulQA as TruthfulQA
from .human_eval.human_eval import HumanEval as HumanEval
from .squad.squad import SQuAD as SQuAD
from .gsm8k.gsm8k import GSM8K as GSM8K
from .math_qa.math_qa import MathQA as MathQA
from .logi_qa.logi_qa import LogiQA as LogiQA
from .bool_q.bool_q import BoolQ as BoolQ
from .arc.arc import ARC as ARC
from .bbq.bbq import BBQ as BBQ
from .lambada.lambada import LAMBADA as LAMBADA
from .winogrande.winogrande import Winogrande as Winogrande
from .equity_med_qa.equity_med_qa import EquityMedQA as EquityMedQA
from .ifeval.ifeval import IFEval as IFEval

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
]
