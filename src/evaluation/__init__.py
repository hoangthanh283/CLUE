"""
Evaluation metrics and procedures for Information Extraction
"""


class Evaluator:
    """Base evaluator class"""

    def __init__(self, config):
        self.config = config

    def evaluate(self, model, test_data):
        """Evaluate model on test data"""
        raise NotImplementedError


class SequenceLabelingEvaluator(Evaluator):
    """Evaluator for sequence labeling tasks"""

    def evaluate(self, model, test_data):
        # TODO: Implement sequence labeling evaluation
        pass

    def compute_f1(self, predictions, targets):
        # TODO: Implement F1 score calculation
        pass

    def compute_precision_recall(self, predictions, targets):
        # TODO: Implement precision and recall
        pass
