from .open import OpenEvaluator
from .qa import QAEvaluator
from .harm import HarmEvaluator
from .ifeval import IFEvaluator

evaluator_mapping = {
    'harm': HarmEvaluator,
    'qa': QAEvaluator,
    'open': OpenEvaluator,
    'ifeval': IFEvaluator,
}