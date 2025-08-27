"""
Model implementations and wrappers for Information Extraction
"""


class BaseModel:
    """Base class for IE models"""

    def __init__(self, config):
        self.config = config

    def forward(self, inputs):
        """Forward pass"""
        raise NotImplementedError

    def save(self, path):
        """Save model"""
        raise NotImplementedError

    def load(self, path):
        """Load model"""
        raise NotImplementedError


class BERTModel(BaseModel):
    """BERT-based model for sequence labeling"""

    def __init__(self, config):
        super().__init__(config)
        # TODO: Initialize BERT model

    def forward(self, inputs):
        # TODO: Implement forward pass
        pass


class BiLSTMModel(BaseModel):
    """BiLSTM model for sequence labeling"""

    def __init__(self, config):
        super().__init__(config)
        # TODO: Initialize BiLSTM model

    def forward(self, inputs):
        # TODO: Implement forward pass
        pass
