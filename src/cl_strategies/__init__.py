"""Continual Learning strategy package.

Public API re-exports core classes from submodules.
"""

from src.cl_strategies.agem import AGEM
from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.er import ExperienceReplay
from src.cl_strategies.er import ExperienceReplay as ReplayStrategy
from src.cl_strategies.ewc import EWC
from src.cl_strategies.ewc import EWC as EWCStrategy
from src.cl_strategies.gem import GEM
from src.cl_strategies.lwf import LwF
from src.cl_strategies.sequential import SequentialFineTuning
from src.cl_strategies.sequential import SequentialFineTuning as NoCLStrategy

__all__ = [
    "BaseCLStrategy",
    "SequentialFineTuning",
    "ExperienceReplay",
    "EWC",
    "LwF",
    "AGEM",
    "GEM",
    # Aliases
    "NoCLStrategy",
    "ReplayStrategy",
    "EWCStrategy",
]
