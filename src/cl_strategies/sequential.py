"""Sequential fine-tuning (baseline)."""

from src.cl_strategies.base import BaseCLStrategy


class SequentialFineTuning(BaseCLStrategy):
    """No continual learning regularization or memory."""

    pass
