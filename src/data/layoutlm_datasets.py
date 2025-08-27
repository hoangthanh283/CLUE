"""
Data loaders and processors for LayoutLM-based Information Extraction
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (LayoutLMTokenizer, LayoutLMv2Tokenizer,
                          LayoutLMv3Tokenizer)

logger = logging.getLogger(__name__)


@dataclass
class DocumentExample:
    """Single document example for LayoutLM models"""

    words: List[str]
    bboxes: List[List[int]]  # [x0, y0, x1, y1] format
    labels: List[str]
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config["dataset"]["name"]
        self.hf_dataset_name = config["dataset"]["hf_dataset_name"]
        self.task_type = config["dataset"]["task_type"]

        # Initialize tokenizer based on model type
        model_name = config["model"]["pretrained_model_name"]
        if "layoutlmv3" in model_name.lower():
            self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name)
        elif "layoutlmv2" in model_name.lower():
            self.tokenizer = LayoutLMv2Tokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

        self.max_seq_length = config["dataset"]["preprocessing"]["max_seq_length"]
        self.image_size = config["dataset"]["preprocessing"]["image_size"]
        self.normalize_bbox = config["dataset"]["preprocessing"]["normalize_bbox"]

    @abstractmethod
    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load and return train, test, and optional validation datasets"""
        pass

    @abstractmethod
    def get_label_list(self) -> List[str]:
        """Get the list of labels for the dataset"""
        pass

    @abstractmethod
    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert HuggingFace dataset to DocumentExample format"""
        pass

    def normalize_bbox_coordinates(self, bbox: List[int], width: int, height: int) -> List[int]:
        """Normalize bbox coordinates to [0, 1000] range"""
        x0, y0, x1, y1 = bbox
        x0 = int(1000 * x0 / width)
        y0 = int(1000 * y0 / height)
        x1 = int(1000 * x1 / width)
        y1 = int(1000 * y1 / height)
        return [x0, y0, x1, y1]

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for LayoutLM processing"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(self.image_size)
        return image


class FUNSDDatasetLoader(BaseDatasetLoader):
    """Loader for FUNSD dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION",
                           "I-QUESTION", "B-ANSWER", "I-ANSWER"]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load FUNSD dataset from HuggingFace"""
        logger.info(f"Loading FUNSD dataset from {self.hf_dataset_name}")

        dataset = load_dataset(self.hf_dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # FUNSD doesn't have a validation split, so we'll create one
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get FUNSD label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert FUNSD dataset to DocumentExample format"""
        examples = []

        for item in dataset:
            words = item["words"]
            bboxes = item["bboxes"]
            ner_tags = item["ner_tags"]
            image = item["image"]

            # Convert numeric labels to string labels
            labels = [self.id2label[tag] for tag in ner_tags]

            # Normalize bboxes if required
            if self.normalize_bbox and image is not None:
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]

            # Prepare image
            if image is not None:
                image = self.prepare_image(image)

            example = DocumentExample(
                words=words,
                bboxes=bboxes,
                labels=labels,
                image=image
            )
            examples.append(example)

        return examples


class CORDDatasetLoader(BaseDatasetLoader):
    """Loader for CORD dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # CORD has 30 entity types + O
        self.label_list = self._get_cord_labels()
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def _get_cord_labels(self) -> List[str]:
        """Get CORD label list"""
        return [
            "O",
            "B-MENU.CNT", "I-MENU.CNT",
            "B-MENU.DISCOUNTPRICE", "I-MENU.DISCOUNTPRICE",
            "B-MENU.ETC", "I-MENU.ETC",
            "B-MENU.ITEMSUBTOTAL", "I-MENU.ITEMSUBTOTAL",
            "B-MENU.NM", "I-MENU.NM",
            "B-MENU.NUM", "I-MENU.NUM",
            "B-MENU.PRICE", "I-MENU.PRICE",
            "B-MENU.SUB_CNT", "I-MENU.SUB_CNT",
            "B-MENU.SUB_ETC", "I-MENU.SUB_ETC",
            "B-MENU.SUB_NM", "I-MENU.SUB_NM",
            "B-MENU.SUB_PRICE", "I-MENU.SUB_PRICE",
            "B-MENU.UNITPRICE", "I-MENU.UNITPRICE",
            "B-SUB_TOTAL.DISCOUNT_PRICE", "I-SUB_TOTAL.DISCOUNT_PRICE",
            "B-SUB_TOTAL.ETC", "I-SUB_TOTAL.ETC",
            "B-SUB_TOTAL.SERVICE_PRICE", "I-SUB_TOTAL.SERVICE_PRICE",
            "B-SUB_TOTAL.SUBTOTAL_PRICE", "I-SUB_TOTAL.SUBTOTAL_PRICE",
            "B-SUB_TOTAL.TAX_PRICE", "I-SUB_TOTAL.TAX_PRICE",
            "B-TOTAL.CASHPRICE", "I-TOTAL.CASHPRICE",
            "B-TOTAL.CREDITCARDPRICE", "I-TOTAL.CREDITCARDPRICE",
            "B-TOTAL.EMONEYPRICE", "I-TOTAL.EMONEYPRICE",
            "B-TOTAL.MENUQTY_CNT", "I-TOTAL.MENUQTY_CNT",
            "B-TOTAL.MENUTYPE_CNT", "I-TOTAL.MENUTYPE_CNT",
            "B-TOTAL.TOTAL_ETC", "I-TOTAL.TOTAL_ETC",
            "B-TOTAL.TOTAL_PRICE", "I-TOTAL.TOTAL_PRICE"
        ]

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load CORD dataset from HuggingFace"""
        logger.info(f"Loading CORD dataset from {self.hf_dataset_name}")

        dataset = load_dataset(self.hf_dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = dataset.get("validation", None)

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get CORD label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert CORD dataset to DocumentExample format"""
        examples = []

        for item in dataset:
            words = item["words"]
            bboxes = item["bboxes"]
            ner_tags = item["ner_tags"]
            image = item["image"]

            # Convert numeric labels to string labels
            labels = [self.id2label[tag] for tag in ner_tags]

            # Normalize bboxes if required
            if self.normalize_bbox and image is not None:
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]

            # Prepare image
            if image is not None:
                image = self.prepare_image(image)

            example = DocumentExample(
                words=words,
                bboxes=bboxes,
                labels=labels,
                image=image
            )
            examples.append(example)

        return examples


class LayoutLMDataset(Dataset):
    """PyTorch Dataset for LayoutLM models"""

    def __init__(
        self,
        examples: List[DocumentExample],
        tokenizer: Union[LayoutLMTokenizer, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
        label2id: Dict[str, int],
        max_seq_length: int = 512,
        include_image: bool = True
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.include_image = include_image

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(
            example.words,
            boxes=example.bboxes,
            word_labels=example.labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # Prepare output
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "bbox": encoding["bbox"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }

        # Add image if available and required
        if self.include_image and example.image is not None:
            # Convert PIL image to tensor
            image_tensor = torch.tensor(np.array(example.image)).permute(2, 0, 1).float() / 255.0
            item["image"] = image_tensor

        return item


def create_data_loader(
    examples: List[DocumentExample],
    tokenizer: Union[LayoutLMTokenizer, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
    label2id: Dict[str, int],
    config: Dict[str, Any],
    is_training: bool = True
) -> DataLoader:
    """Create DataLoader for LayoutLM training/evaluation"""

    dataset = LayoutLMDataset(
        examples=examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_length=config["dataset"]["preprocessing"]["max_seq_length"],
        include_image=config["dataset"]["preprocessing"]["include_image"]
    )

    batch_size = config["training"]["batch_size"]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training and config["data_processing"]["shuffle_train"],
        num_workers=4,
        pin_memory=True
    )


# Dataset factory
DATASET_LOADERS = {
    "funsd": FUNSDDatasetLoader,
    "cord": CORDDatasetLoader,
}

# Import additional datasets
try:
    from .additional_datasets import ADDITIONAL_DATASET_LOADERS
    DATASET_LOADERS.update(ADDITIONAL_DATASET_LOADERS)
except ImportError:
    logger.warning("Additional datasets not available")


def get_dataset_loader(config: Dict[str, Any]) -> BaseDatasetLoader:
    """Factory function to get appropriate dataset loader"""
    dataset_name = config["dataset"]["name"].lower()

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Supported datasets: {list(DATASET_LOADERS.keys())}")

    return DATASET_LOADERS[dataset_name](config)
