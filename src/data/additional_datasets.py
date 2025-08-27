"""
Additional dataset loaders for LayoutLM-based Information Extraction
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset
from datasets import load_dataset

from .layoutlm_datasets import BaseDatasetLoader, DocumentExample

logger = logging.getLogger(__name__)


class SROIEDatasetLoader(BaseDatasetLoader):
    """Loader for SROIE dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.label_list = [
            "O",
            "B-COMPANY", "I-COMPANY",
            "B-DATE", "I-DATE",
            "B-ADDRESS", "I-ADDRESS",
            "B-TOTAL", "I-TOTAL"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load SROIE dataset from HuggingFace"""
        logger.info(f"Loading SROIE dataset from {self.hf_dataset_name}")

        dataset = load_dataset(self.hf_dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Create validation split if needed
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get SROIE label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert SROIE dataset to DocumentExample format"""
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


class XFUNDDatasetLoader(BaseDatasetLoader):
    """Loader for XFUND dataset (multilingual FUNSD)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # XFUND uses same labels as FUNSD but supports multiple languages
        self.label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION",
                           "I-QUESTION", "B-ANSWER", "I-ANSWER"]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # Language code for XFUND (e.g., "xfund.zh", "xfund.ja", etc.)
        self.language = config["dataset"].get("language", "zh")

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load XFUND dataset from HuggingFace"""
        # XFUND dataset name includes language code
        dataset_name = f"{self.hf_dataset_name}.{self.language}"
        logger.info(f"Loading XFUND dataset from {dataset_name}")

        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]  # XFUND uses 'validation' as test set

        # Create validation split from training data
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get XFUND label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert XFUND dataset to DocumentExample format"""
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


class WildReceiptDatasetLoader(BaseDatasetLoader):
    """Loader for WildReceipt dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # WildReceipt has 26 entity types
        self.label_list = [
            "O",
            "B-Store_name_key", "I-Store_name_key",
            "B-Store_name_value", "I-Store_name_value",
            "B-Store_addr_key", "I-Store_addr_key",
            "B-Store_addr_value", "I-Store_addr_value",
            "B-Tel_key", "I-Tel_key",
            "B-Tel_value", "I-Tel_value",
            "B-Date_key", "I-Date_key",
            "B-Date_value", "I-Date_value",
            "B-Time_key", "I-Time_key",
            "B-Time_value", "I-Time_value",
            "B-Prod_item_key", "I-Prod_item_key",
            "B-Prod_item_value", "I-Prod_item_value",
            "B-Prod_quantity_key", "I-Prod_quantity_key",
            "B-Prod_quantity_value", "I-Prod_quantity_value",
            "B-Prod_price_key", "I-Prod_price_key",
            "B-Prod_price_value", "I-Prod_price_value",
            "B-Total_key", "I-Total_key",
            "B-Total_value", "I-Total_value",
            "B-Others", "I-Others"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load WildReceipt dataset from HuggingFace"""
        logger.info(f"Loading WildReceipt dataset from {self.hf_dataset_name}")

        dataset = load_dataset(self.hf_dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Create validation split if needed
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get WildReceipt label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert WildReceipt dataset to DocumentExample format"""
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


# Update the dataset factory with new loaders
ADDITIONAL_DATASET_LOADERS = {
    "sroie": SROIEDatasetLoader,
    "xfund": XFUNDDatasetLoader,
    "wildreceipt": WildReceiptDatasetLoader,
}
