"""
LayoutLM model implementations for Information Extraction
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from seqeval.scheme import IOB2
from transformers import (LayoutLMConfig, LayoutLMModel, LayoutLMv2Config,
                          LayoutLMv2Model, LayoutLMv3Config, LayoutLMv3Model)

logger = logging.getLogger(__name__)


class BaseLayoutLMModel(nn.Module, ABC):
    """Base class for LayoutLM models"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_config = config["model"]
        self.num_labels = self.model_config["config"]["num_labels"]

        # Initialize the backbone model
        self.backbone = self._initialize_backbone()

        # Classification head
        self.dropout = nn.Dropout(
            self.model_config["config"].get("classifier_dropout", 0.1)
        )
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)

        # Initialize weights
        self._init_weights()

    @abstractmethod
    def _initialize_backbone(self):
        """Initialize the backbone LayoutLM model"""
        pass

    def _init_weights(self):
        """Initialize classification head weights"""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # For LayoutLMv3
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""

        # Prepare inputs for the backbone model
        backbone_inputs = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
        }

        # Add optional inputs if they exist
        if token_type_ids is not None:
            backbone_inputs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            backbone_inputs["position_ids"] = position_ids
        if head_mask is not None:
            backbone_inputs["head_mask"] = head_mask

        # Handle visual inputs for different LayoutLM versions
        if hasattr(self.backbone, 'config') and 'layoutlmv3' in self.backbone.config.model_type:
            # LayoutLMv3 - use pixel_values if provided, otherwise text+layout only
            if pixel_values is not None:
                backbone_inputs["pixel_values"] = pixel_values
        else:
            # LayoutLMv1/v2 - use image if provided
            if image is not None:
                backbone_inputs["image"] = image

        # Forward through backbone
        outputs = self.backbone(**backbone_inputs)
        sequence_output = outputs.last_hidden_state

        # Apply dropout and classification
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }

    def save_pretrained(self, save_directory: str):
        """Save the model"""
        self.backbone.save_pretrained(save_directory)
        # Save the classification head separately
        torch.save(self.classifier.state_dict(), f"{save_directory}/classifier.pt")

    def load_pretrained(self, load_directory: str):
        """Load the model"""
        self.backbone = self.backbone.from_pretrained(load_directory)
        # Load the classification head
        classifier_path = f"{load_directory}/classifier.pt"
        # Load to CPU first; caller can move model to desired device afterward
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

    def reset_classifier(self, num_labels: int):
        """Reset the classification head to a new label count (for sequential FT).

        Keeps the backbone intact; reinitializes a new Linear layer.
        """
        self.num_labels = int(num_labels)
        device = next(self.parameters()).device
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels).to(device)
        self._init_weights()


class LayoutLMForTokenClassification(BaseLayoutLMModel):
    """LayoutLM v1 model for token classification"""

    def _initialize_backbone(self):
        """Initialize LayoutLM v1 backbone"""
        model_name = self.model_config["pretrained_model_name"]

        # Load configuration
        config = LayoutLMConfig.from_pretrained(model_name)

        # Update config with custom settings
        config.hidden_dropout_prob = self.model_config["config"].get(
            "hidden_dropout_prob", 0.1
        )
        config.attention_probs_dropout_prob = self.model_config["config"].get(
            "attention_probs_dropout_prob", 0.1
        )

        # Initialize model
        model = LayoutLMModel.from_pretrained(model_name, config=config)

        logger.info(f"Initialized LayoutLM v1 from {model_name}")
        return model


class LayoutLMv2ForTokenClassification(BaseLayoutLMModel):
    """LayoutLM v2 model for token classification"""

    def _initialize_backbone(self):
        """Initialize LayoutLM v2 backbone"""
        model_name = self.model_config["pretrained_model_name"]

        # Load configuration
        config = LayoutLMv2Config.from_pretrained(model_name)

        # Update config with custom settings
        config.hidden_dropout_prob = self.model_config["config"].get(
            "hidden_dropout_prob", 0.1
        )
        config.attention_probs_dropout_prob = self.model_config["config"].get(
            "attention_probs_dropout_prob", 0.1
        )

        # Initialize model
        model = LayoutLMv2Model.from_pretrained(model_name, config=config)

        logger.info(f"Initialized LayoutLM v2 from {model_name}")
        return model


class LayoutLMv3ForTokenClassification(BaseLayoutLMModel):
    """LayoutLM v3 model for token classification"""

    def _initialize_backbone(self):
        """Initialize LayoutLM v3 backbone"""
        model_name = self.model_config["pretrained_model_name"]

        # Load configuration
        config = LayoutLMv3Config.from_pretrained(model_name)

        # Update config with custom settings
        config.hidden_dropout_prob = self.model_config["config"].get(
            "hidden_dropout_prob", 0.1
        )
        config.attention_probs_dropout_prob = self.model_config["config"].get(
            "attention_probs_dropout_prob", 0.1
        )

        # Initialize model
        model = LayoutLMv3Model.from_pretrained(model_name, config=config)

        logger.info(f"Initialized LayoutLM v3 from {model_name}")
        return model


class LayoutLMMetrics:
    """Metrics computation for LayoutLM token classification"""

    def __init__(self, label_list: List[str], id2label: Dict[int, str]):
        self.label_list = label_list
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute token-level and entity-level metrics"""
        from sklearn.metrics import (accuracy_score,
                                     precision_recall_fscore_support)

        # Convert predictions and labels to numpy
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        attention_mask = attention_mask.detach().cpu().numpy()

        # Get predictions
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (subword tokens and padding)
        true_predictions = []
        true_labels = []

        for prediction, label, mask in zip(predictions, labels, attention_mask):
            for pred, lab, m in zip(prediction, label, mask):
                if m and lab != -100:  # -100 is the ignore index
                    true_predictions.append(pred)
                    true_labels.append(lab)

        # Token-level metrics
        accuracy = accuracy_score(true_labels, true_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average="macro", zero_division=0
        )

        # Entity-level metrics (if using BIO tagging)
        entity_f1 = self._compute_entity_level_f1(true_predictions, true_labels)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "entity_f1": entity_f1
        }

    def _compute_entity_level_f1(self, predictions: List[int], labels: List[int]) -> float:
        """Compute entity-level F1 score for BIO tagging"""
        try:
            # Convert to label strings.
            from seqeval.metrics import f1_score
            pred_labels = [self.id2label[pred] for pred in predictions]
            true_labels = [self.id2label[label] for label in labels]
            return f1_score([true_labels], [pred_labels], mode="strict", scheme=IOB2)
        except ImportError:
            logger.warning("seqeval not available, using token-level F1 as entity F1")
            from sklearn.metrics import f1_score
            return f1_score(labels, predictions, average="macro", zero_division=0)


# Model factory
MODEL_CLASSES = {
    "layoutlmv3-base": LayoutLMv3ForTokenClassification,
    "layoutlmv3": LayoutLMv3ForTokenClassification,
    "layoutlmv2-base-uncased": LayoutLMv2ForTokenClassification,
    "layoutlmv2": LayoutLMv2ForTokenClassification,
    "layoutlm-base-uncased": LayoutLMForTokenClassification,
    "layoutlm": LayoutLMForTokenClassification,
}


def get_model(config: Dict[str, Any]) -> BaseLayoutLMModel:
    """Factory function to get appropriate LayoutLM model"""
    model_type = config["model"]["model_type"].lower()

    # Handle model type variations
    for key in MODEL_CLASSES.keys():
        if key in model_type:
            model_class = MODEL_CLASSES[key]
            break
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_CLASSES.keys())}")

    logger.info(f"Creating {model_class.__name__} with config: {config['model']}")
    return model_class(config)
