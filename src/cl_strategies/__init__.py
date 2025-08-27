"""
Continual Learning Strategies for Information Extraction
"""


class BaseCLStrategy:
    """Base class for continual learning strategies"""

    def __init__(self, config):
        self.config = config

    def before_task(self, model, task_id):
        """Called before training on a new task"""
        pass

    def after_task(self, model, task_id):
        """Called after training on a task"""
        pass

    def compute_loss(self, model, outputs, targets):
        """Compute loss with CL regularization"""
        raise NotImplementedError


class EWCStrategy(BaseCLStrategy):
    """Elastic Weight Consolidation"""

    def compute_loss(self, model, outputs, targets):
        # TODO: Implement EWC loss
        pass


class ReplayStrategy(BaseCLStrategy):
    """Experience Replay"""

    def compute_loss(self, model, outputs, targets):
        # TODO: Implement replay loss
        pass


class NoCLStrategy(BaseCLStrategy):
    """No continual learning (baseline)"""

    def compute_loss(self, model, outputs, targets):
        # TODO: Implement standard loss
        pass
