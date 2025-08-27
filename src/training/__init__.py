"""
Training procedures, losses, and optimizers
"""


class Trainer:
    """Base trainer class"""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, train_data):
        """Training loop"""
        raise NotImplementedError

    def validate(self, val_data):
        """Validation loop"""
        raise NotImplementedError


class StandardTrainer(Trainer):
    """Standard training procedure"""

    def train(self, train_data):
        # TODO: Implement standard training
        pass


class ContinualTrainer(Trainer):
    """Training with continual learning strategies"""

    def __init__(self, model, config, cl_strategy):
        super().__init__(model, config)
        self.cl_strategy = cl_strategy

    def train(self, train_data):
        # TODO: Implement continual training
        pass
