"""
Dataset loaders and processors for Information Extraction tasks
"""


class DatasetLoader:
    """Base class for dataset loaders"""

    def __init__(self, config):
        self.config = config

    def load_data(self):
        """Load and return dataset"""
        raise NotImplementedError

    def preprocess(self, data):
        """Preprocess the data"""
        raise NotImplementedError


class CoNLLDatasetLoader(DatasetLoader):
    """Loader for CoNLL format datasets"""

    def load_data(self):
        # TODO: Implement CoNLL data loading
        pass


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom dataset formats"""

    def load_data(self):
        # TODO: Implement custom data loading
        pass
