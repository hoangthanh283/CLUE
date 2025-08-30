"""
Data loaders and processors for LayoutLM-based Information Extraction.
"""

import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (LayoutLMTokenizerFast, LayoutLMv2Tokenizer,
                          LayoutLMv3Tokenizer)

from src.data.label_space import (UNIFIED_LABEL_LIST, map_cord_entity,
                                  map_form_entity, map_sroie_entity,
                                  map_wildreceipt_entity)

logger = logging.getLogger(__name__)


@dataclass
class DocumentExample:
    """Single document example for LayoutLM models"""

    words: List[str]
    bboxes: List[List[int]]  # [x0, y0, x1, y1] format.
    labels: List[str]


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config["dataset"]["name"]
        self.hf_dataset_name = config["dataset"]["hf_dataset_name"]
        self.task_type = config["dataset"]["task_type"]
        # Streaming mode removed; always use on-demand map-style datasets
        # Cache for actual dataset lengths (will be populated during load_data).
        self._dataset_lengths = {}
        # Cache image dimensions to avoid repeated disk I/O.
        self._image_size_cache: Dict[str, Tuple[int, int]] = {}

        # Initialize tokenizer based on model type
        model_name = config["model"]["pretrained_model_name"]
        if "layoutlmv3" in model_name.lower():
            self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name)
        elif "layoutlmv2" in model_name.lower():
            self.tokenizer = LayoutLMv2Tokenizer.from_pretrained(model_name)
        else:
            # Use Fast tokenizer for LayoutLM v1 to support boxes/labels alignment.
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)

        self.max_seq_length = config["dataset"]["preprocessing"]["max_seq_length"]
        self.normalize_bbox = config["dataset"]["preprocessing"]["normalize_bbox"]
        self.use_unified_labels = bool(config.get("label_space", {}).get("unified", False))

    @abstractmethod
    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load and return train, test, and optional validation datasets"""
        pass

    @abstractmethod
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single dataset item into DocumentExample format"""
        pass

    def _compute_dataset_length(self, dataset_name: str, hf_dataset_name: str, split_name: str, quick_mode: bool = True
                                ) -> int:
        """Compute actual dataset length by loading the dataset

        Args:
            dataset_name: Name of the dataset
            hf_dataset_name: HuggingFace dataset name
            split_name: Split name (train, test, validation)
            quick_mode: Reserved for compatibility (no special behavior)

        Returns:
            Actual dataset length or 0 if quick_mode is enabled
        """
        cache_key = f"{dataset_name}_{split_name}"
        if cache_key in self._dataset_lengths:
            return self._dataset_lengths[cache_key]

        try:
            logger.info(f"Computing length for {dataset_name} {split_name} split...")
            # Load dataset to get length
            non_streaming_dataset = load_dataset(hf_dataset_name, streaming=False)

            # Handle different dataset split mappings
            actual_split_name = split_name
            if dataset_name == "xfund" and split_name == "test":
                actual_split_name = "val"  # XFUND uses "val" for test split
            elif split_name == "validation":
                # For validation splits created from train split, get the train length and compute validation length
                # based on validation_split ratio
                if "train" in non_streaming_dataset:
                    train_length = len(non_streaming_dataset["train"])
                    validation_split = self.config["data_processing"].get("validation_split", 0.1)

                    # Apply language filtering for XFUND
                    if dataset_name == "xfund":
                        language = self.config["dataset"].get("language", "zh")
                        train_filtered = []
                        for example in non_streaming_dataset["train"]:
                            if example["id"].startswith(f"{language}_"):
                                train_filtered.append(example)
                        train_length = len(train_filtered)

                    if isinstance(validation_split, float) and 0 < validation_split < 1:
                        val_length = int(train_length * validation_split)
                        self._dataset_lengths[cache_key] = val_length
                        logger.info(f"Cached validation length for {cache_key}: {val_length} (from train: "
                                    f"{train_length})")
                        return val_length
                    else:
                        return 0
                else:
                    logger.warning("No train split found to compute validation length")
                    return 0

            if actual_split_name in non_streaming_dataset:
                length = len(non_streaming_dataset[actual_split_name])
                # Apply language filtering for XFUND.
                if dataset_name == "xfund":
                    language = self.config["dataset"].get("language", "zh")
                    filtered_dataset = []
                    for example in non_streaming_dataset[actual_split_name]:
                        if example["id"].startswith(f"{language}_"):
                            filtered_dataset.append(example)
                    length = len(filtered_dataset)

                self._dataset_lengths[cache_key] = length
                logger.info(f"Cached length for {cache_key}: {length}")
                return length
            else:
                logger.warning(f"Split {actual_split_name} not found in dataset {hf_dataset_name}")
                return 0
        except Exception as error:
            logger.warning(f"Could not compute length for {cache_key}: {error}")
            return 0

    def get_image_dimensions(self, image) -> Tuple[int, int]:
        """Get image dimensions without loading full image for processing

        Args:
            image: Image in various formats (PIL Image, dict with bytes, etc.)

        Returns:
            Tuple of (width, height)
        """
        try:
            # If it's already a PIL Image, get dimensions directly
            if isinstance(image, Image.Image):
                return image.size

            # If it's a dictionary (e.g., with raw bytes), try to extract bytes
            if isinstance(image, dict):
                # Prefer caching by path
                if "path" in image and isinstance(image["path"], str):
                    key = image["path"]
                    if key in self._image_size_cache:
                        return self._image_size_cache[key]
                    with Image.open(image["path"]) as img:
                        size = img.size
                    self._image_size_cache[key] = size
                    return size
                elif "bytes" in image and isinstance(image["bytes"], (bytes, bytearray)):
                    raw = image["bytes"]
                    # Lightweight cache key derived from length and prefix to avoid hashing entire file
                    prefix = bytes(raw[:64])
                    key = f"bytes:{len(raw)}:{prefix}"
                    if key in self._image_size_cache:
                        return self._image_size_cache[key]
                    with Image.open(io.BytesIO(raw)) as img:
                        size = img.size
                    self._image_size_cache[key] = size
                    return size
                else:
                    logger.warning(f"Unknown image dict format: {list(image.keys())}")
                    return 1000, 1000  # Fallback

            # If it's bytes directly
            if isinstance(image, bytes):
                prefix = image[:64]
                key = f"bytes:{len(image)}:{prefix}"
                if key in self._image_size_cache:
                    return self._image_size_cache[key]
                with Image.open(io.BytesIO(image)) as img:
                    size = img.size
                self._image_size_cache[key] = size
                return size

            # If it's a file path
            if isinstance(image, str):
                if image in self._image_size_cache:
                    return self._image_size_cache[image]
                with Image.open(image) as img:
                    size = img.size
                self._image_size_cache[image] = size
                return size

            # If it has a size attribute, try to use it
            if hasattr(image, "size"):
                return image.size

            logger.warning(f"Unknown image type: {type(image)}. Using default dimensions.")
            return 1000, 1000  # Fallback
        except Exception as e:
            logger.warning(f"Error getting image dimensions: {e}. Using default dimensions.")
            return 1000, 1000  # Fallback

    def normalize_bbox_coordinates(self, bbox: List[int], width: int, height: int) -> List[int]:
        """Normalize bbox coordinates to [0, 1000] range and clamp to valid bounds"""
        x0, y0, x1, y1 = bbox
        x0 = int(1000 * x0 / width)
        y0 = int(1000 * y0 / height)
        x1 = int(1000 * x1 / width)
        y1 = int(1000 * y1 / height)

        # Clamp values to [0, 1000] range as required by LayoutLMv3
        x0 = max(0, min(1000, x0))
        y0 = max(0, min(1000, y0))
        x1 = max(0, min(1000, x1))
        y1 = max(0, min(1000, y1))
        return [x0, y0, x1, y1]

    def get_dataset_length(self, split_name: str) -> int:
        """Get the actual length of a dataset split

        Args:
            split_name: Split name (train, test, validation)

        Returns:
            Actual dataset length
        """
        return self._compute_dataset_length(self.dataset_name, self.hf_dataset_name, split_name)

    def get_label_list(self) -> List[str]:
        return UNIFIED_LABEL_LIST if self.use_unified_labels and UNIFIED_LABEL_LIST else self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert dataset to DocumentExample format."""
        examples: List[DocumentExample] = []
        for item in dataset:
            example = self._process_single_item(item)
            if not example:
                continue
            examples.append(example)
        return examples


class LayoutLMDataset(Dataset):
    """Dataset that reads HF rows on-demand and tokenizes lazily."""

    def __init__(
        self,
        dataset_loader: BaseDatasetLoader,
        hf_dataset: HFDataset,
        tokenizer: Union[LayoutLMTokenizerFast, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
        label2id: Dict[str, int],
        max_seq_length: int = 512,
    ):
        self.dataset_loader = dataset_loader
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.hf_dataset[int(idx)]
        example = self.dataset_loader._process_single_item(item)
        integer_labels = [self.label2id.get(label, 0) for label in example.labels]
        encoding = self.tokenizer(
            example.words,
            boxes=example.bboxes,
            word_labels=integer_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "bbox": encoding["bbox"].squeeze(),
            "labels": encoding["labels"].squeeze(),
        }


class FUNSDDatasetLoader(BaseDatasetLoader):
    """DatasetLoader for FUNSD dataset"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hf_dataset_name = config["dataset"]["hf_dataset_name"]
        self.label_list = [
            "O",
            "B-HEADER", "I-HEADER",
            "B-QUESTION", "I-QUESTION",
            "B-ANSWER", "I-ANSWER",
        ]
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load FUNSD dataset from HuggingFace"""
        logger.info(f"Loading FUNSD dataset from {self.hf_dataset_name}")
        dataset = load_dataset(self.hf_dataset_name, streaming=False)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = dataset.get("validation", None)

        # Create validation split if we don't have one and validation_split is configured.
        if val_dataset is None:
            validation_split = self.config["data_processing"].get("validation_split", 0.1)
            if isinstance(validation_split, float) and 0 < validation_split < 1:
                # Use built-in split for map-style dataset.
                train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = train_val["train"]
                val_dataset = train_val["test"]
        return train_dataset, test_dataset, val_dataset

    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single FUNSD dataset item into DocumentExample format"""
        words = item["words"]
        if "bboxes" in item:
            bboxes = item["bboxes"]
        elif "actual_boxes" in item:
            bboxes = item["actual_boxes"]
            if "page_width" in item and "page_height" in item:
                width, height = item["page_width"], item["page_height"]
            elif "image" in item:
                # Get dimensions from image if page dimensions not available.
                width, height = self.get_image_dimensions(item["image"])
            else:
                # Use reasonable defaults for SROIE documents.
                raise ValueError("No page dimensions or image found in SROIE item.")
            bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]
        else:
            raise ValueError(f"Unknown SROIE dataset format. Available keys: {list(item.keys())}")

        # Convert numeric labels to string labels.
        ner_tags = item["ner_tags"]
        labels = [self.id2label[tag] for tag in ner_tags]
        if self.use_unified_labels and UNIFIED_LABEL_LIST:
            # Map labels to unified space if enabled.
            mapped: List[str] = []
            for lab in labels:
                if lab == "O":
                    mapped.append("O")
                    continue
                prefix, ent = lab.split("-", 1)
                uent = map_form_entity(ent)
                mapped.append(f"{prefix}-{uent}" if uent else "O")
            labels = mapped
        return DocumentExample(words=words, bboxes=bboxes, labels=labels)


class CORDDatasetLoader(BaseDatasetLoader):
    """Loader for CORD dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # CORD has 30 entity types + O.
        self.label_list = self._get_cord_labels()
        self.label2id = {label: ii for ii, label in enumerate(self.label_list)}
        self.id2label = {ii: label for ii, label in enumerate(self.label_list)}

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
        dataset = load_dataset(self.hf_dataset_name, streaming=False)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = dataset.get("validation", None)

        # Create validation split if we don't have one and validation_split is configured.
        if val_dataset is None:
            validation_split = self.config["data_processing"].get("validation_split", 0.1)
            if isinstance(validation_split, float) and 0 < validation_split < 1:
                train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = train_val["train"]
                val_dataset = train_val["test"]
        return train_dataset, test_dataset, val_dataset

    def _process_single_item(self, item: Dict[str, Any]) -> Optional[DocumentExample]:
        """Process a single CORD dataset item into DocumentExample format"""
        ground_truth = json.loads(item["ground_truth"])
        words: List[str] = []
        bboxes: List[List[int]] = []
        labels: List[str] = []
        for line in ground_truth.get("valid_line", []):
            category = line.get("category", "other")
            # Map CORD categories to our label format.
            if category.upper() not in [label.replace("B-", "").replace("I-", "") for label in self.label_list]:
                category = "other"

            line_words = line.get("words", [])
            for idx, word_info in enumerate(line_words):
                words.append(word_info["text"])
                # Convert quad to bbox [x0, y0, x1, y1].
                quad = word_info["quad"]
                x0 = min(quad["x1"], quad["x2"], quad["x3"], quad["x4"])
                y0 = min(quad["y1"], quad["y2"], quad["y3"], quad["y4"])
                x1 = max(quad["x1"], quad["x2"], quad["x3"], quad["x4"])
                y1 = max(quad["y1"], quad["y2"], quad["y3"], quad["y4"])
                bboxes.append([x0, y0, x1, y1])
                # Assign B-/I- labels.
                if idx == 0:
                    labels.append(f"B-{category.upper()}")
                else:
                    labels.append(f"I-{category.upper()}")

        if not words:
            # If no valid words found, skip this example.
            return None

        if self.normalize_bbox:
            # Normalize bboxes if required using actual image dimensions.
            if "image" in item:
                width, height = self.get_image_dimensions(item["image"])
            else:
                logger.warning("No image found in CORD item, using default dimensions")
                width, height = 1000, 1000
            bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]

        if self.use_unified_labels and UNIFIED_LABEL_LIST:
            # Map to unified label space if enabled.
            mapped_labels: List[str] = []
            for lab in labels:
                if lab == "O":
                    mapped_labels.append("O")
                    continue
                prefix, ent = lab.split("-", 1)
                uent = map_cord_entity(ent)
                mapped_labels.append(f"{prefix}-{uent}" if uent else "O")
            labels = mapped_labels
        return DocumentExample(words=words, bboxes=bboxes, labels=labels)


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
        dataset = load_dataset(self.hf_dataset_name, streaming=False)
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

    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single SROIE dataset item into DocumentExample format"""
        words = item["words"]

        # Handle different SROIE dataset formats
        if "bboxes" in item:
            bboxes = item["bboxes"]
        elif "actual_boxes" in item:
            bboxes = item["actual_boxes"]
            if "page_width" in item and "page_height" in item:
                width, height = item["page_width"], item["page_height"]
            elif "image" in item:
                # Get dimensions from image if page dimensions not available.
                width, height = self.get_image_dimensions(item["image"])
            else:
                # Use reasonable defaults for SROIE documents.
                raise ValueError("No page dimensions or image found in SROIE item.")
            bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]
        else:
            raise ValueError(f"Unknown SROIE dataset format. Available keys: {list(item.keys())}")

        labels = [kk.upper() for kk in item["labels"]]
        if self.use_unified_labels and UNIFIED_LABEL_LIST:
            # Map to unified label space if enabled.
            mapped: List[str] = []
            for lab in labels:
                if lab == "O":
                    mapped.append("O")
                    continue
                prefix, ent = lab.split("-", 1)
                uent = map_sroie_entity(ent)
                mapped.append(f"{prefix}-{uent}" if uent else "O")
            labels = mapped
        return DocumentExample(words=words, bboxes=bboxes, labels=labels)


class XFUNDDatasetLoader(BaseDatasetLoader):
    """Loader for XFUND dataset (multilingual FUNSD)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # XFUND uses same labels as FUNSD but supports multiple languages.
        self.label_list = [
            "O",
            "B-HEADER", "I-HEADER",
            "B-QUESTION", "I-QUESTION",
            "B-ANSWER", "I-ANSWER",
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        # Language code for XFUND (e.g., "xfund.zh", "xfund.ja", etc.)
        self.language = config["dataset"].get("language", "zh")

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load XFUND dataset from HuggingFace and filter by language"""
        logger.info(f"Loading XFUND dataset from {self.hf_dataset_name} and filtering by language: {self.language}")
        dataset = load_dataset(self.hf_dataset_name, streaming=False)
        filtered_train: List[Dict[str, Any]] = []
        for item in dataset["train"]:
            if item["id"].startswith(f"{self.language}_"):
                filtered_train.append(item)
        filtered_train = HFDataset.from_list(filtered_train)

        # Create validation split from filtered training data.
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = filtered_train.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            train_dataset = filtered_train
            val_dataset = None

        # Prepare test split separately and filter by language.
        filtered_test: List[Dict[str, Any]] = []
        for item in dataset["val"]:
            if item["id"].startswith(f"{self.language}_"):
                filtered_test.append(item)
        test_dataset = HFDataset.from_list(filtered_test)
        return train_dataset, test_dataset, val_dataset

    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single XFUND dataset item into DocumentExample format"""
        words = item["words"]
        bboxes = item["bboxes"]
        ner_tags = item["ner_tags"]
        # Convert numeric labels to string labels.
        labels = [self.id2label[tag] for tag in ner_tags]

        if self.normalize_bbox:
            # Normalize bboxes if required, but skip if they already look normalized (0..1000).
            if "image" in item:
                width, height = self.get_image_dimensions(item["image"])
            else:
                logger.warning("No image found in XFUND item, using default dimensions")
                width, height = 1000, 1000
            bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]

        if self.use_unified_labels and UNIFIED_LABEL_LIST:
            # Map to unified form entities if enabled.
            mapped: List[str] = []
            for lab in labels:
                if lab == "O":
                    mapped.append("O")
                    continue
                prefix, ent = lab.split("-", 1)
                uent = map_form_entity(ent)
                mapped.append(f"{prefix}-{uent}" if uent else "O")
            labels = mapped
        return DocumentExample(words=words, bboxes=bboxes, labels=labels)


class WildReceiptDatasetLoader(BaseDatasetLoader):
    """Loader for WildReceipt dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # WildReceipt has 26 entity types.
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
        self.label2id = {label: ii for ii, label in enumerate(self.label_list)}
        self.id2label = {ii: label for ii, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load WildReceipt dataset from HuggingFace"""
        logger.info(f"Loading WildReceipt dataset from {self.hf_dataset_name}")
        dataset = load_dataset(self.hf_dataset_name, streaming=False)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Create validation split if needed.
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
        else:
            val_dataset = None
        return train_dataset, test_dataset, val_dataset

    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single WildReceipt dataset item into DocumentExample format"""
        words = item["words"]
        bboxes = item["bboxes"]
        ner_tags = item["ner_tags"]
        # Convert numeric labels to string labels.
        labels = [self.id2label[tag] for tag in ner_tags]
        if self.normalize_bbox:
            # Get actual image dimensions for proper normalization.
            if "image" in item:
                width, height = self.get_image_dimensions(item["image"])
            else:
                logger.warning("No image found in WildReceipt item, using default dimensions")
                width, height = 1000, 1000
            bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]

        # Map to unified label space if enabled.
        if self.use_unified_labels and UNIFIED_LABEL_LIST:
            mapped: List[str] = []
            for lab in labels:
                if lab == "O":
                    mapped.append("O")
                    continue
                prefix, ent = lab.split("-", 1)
                # WildReceipt categories like "Store_name_key" -> RCPT.STORE_NAME.
                uent = map_wildreceipt_entity(ent)
                mapped.append(f"{prefix}-{uent}" if uent else "O")
            labels = mapped
        return DocumentExample(words=words, bboxes=bboxes, labels=labels)


def create_data_loader(
    *,
    tokenizer: Union[LayoutLMTokenizerFast, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
    label2id: Dict[str, int],
    config: Dict[str, Any],
    is_training: bool,
    dataset_loader: BaseDatasetLoader,
    hf_dataset: HFDataset,
) -> DataLoader:
    """Create DataLoader that reads HF rows on-demand (no full preloading)."""
    dataset = LayoutLMDataset(
        dataset_loader=dataset_loader,
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_length=config["dataset"]["preprocessing"]["max_seq_length"],
    )
    workers = int(config["training"].get("num_workers", 4))
    dl_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "shuffle": is_training and config.get("data_processing", {}).get("shuffle_train", False),
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        if "persistent_workers" in config["training"]:
            dl_kwargs["persistent_workers"] = bool(config["training"]["persistent_workers"])
        if "prefetch_factor" in config["training"]:
            dl_kwargs["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    return DataLoader(dataset, **dl_kwargs)


DATASET_LOADERS = {
    "funsd": FUNSDDatasetLoader,
    "cord": CORDDatasetLoader,
    "sroie": SROIEDatasetLoader,
    "xfund": XFUNDDatasetLoader,
    "wildreceipt": WildReceiptDatasetLoader,
}


def get_dataset_loader(config: Dict[str, Any]) -> BaseDatasetLoader:
    """Factory function to get appropriate dataset loader"""
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[dataset_name](config)
