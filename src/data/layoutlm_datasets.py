"""
Data loaders and processors for LayoutLM-based Information Extraction
"""
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (LayoutLMTokenizerFast, LayoutLMv2Tokenizer,
                          LayoutLMv3Tokenizer)

logger = logging.getLogger(__name__)


@dataclass
class DocumentExample:
    """Single document example for LayoutLM models"""

    words: List[str]
    bboxes: List[List[int]]  # [x0, y0, x1, y1] format.
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
        
        # Check if streaming mode is enabled
        self.streaming = config["dataset"].get("streaming", False)
        
        # Cache for actual dataset lengths (will be populated during load_data)
        self._dataset_lengths = {}

        # Initialize tokenizer based on model type
        model_name = config["model"]["pretrained_model_name"]
        if "layoutlmv3" in model_name.lower():
            self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name)
        elif "layoutlmv2" in model_name.lower():
            self.tokenizer = LayoutLMv2Tokenizer.from_pretrained(model_name)
        else:
            # Use Fast tokenizer for LayoutLM v1 to support boxes/labels alignment
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)

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
    
    def create_examples_iterator(self, dataset: HFDataset) -> Iterator[DocumentExample]:
        """Create an iterator over DocumentExample objects for streaming"""
        for item in dataset:
            yield self._process_single_item(item)
    
    @abstractmethod
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single dataset item into DocumentExample format"""
        pass

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

    def prepare_image(self, image) -> Optional[Image.Image]:
        """Prepare image for LayoutLM processing"""
        try:
            # Handle PIL Image
            if hasattr(image, 'mode'):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = image.resize(self.image_size)
                return image
            else:
                # Image might be in a different format (dict, etc.) in streaming mode
                logger.warning(f"Cannot process image of type {type(image)}. Skipping image processing.")
                return None
        except Exception as e:
            logger.warning(f"Error processing image: {e}. Skipping image processing.")
            return None
    
    def _compute_dataset_length(self, dataset_name: str, hf_dataset_name: str, split_name: str) -> int:
        """Compute actual dataset length by loading in non-streaming mode
        
        Args:
            dataset_name: Name of the dataset
            hf_dataset_name: HuggingFace dataset name
            split_name: Split name (train, test, validation)
            
        Returns:
            Actual dataset length
        """
        cache_key = f"{dataset_name}_{split_name}"
        if cache_key in self._dataset_lengths:
            return self._dataset_lengths[cache_key]
        
        try:
            logger.info(f"Computing length for {dataset_name} {split_name} split...")
            # Load in non-streaming mode to get length
            non_streaming_dataset = load_dataset(hf_dataset_name, streaming=False)

            # Handle different dataset split mappings
            actual_split_name = split_name
            if dataset_name == "xfund" and split_name == "test":
                actual_split_name = "val"  # XFUND uses "val" for test split
            elif split_name == "validation":
                # For validation splits created from train split, get the train length 
                # and compute validation length based on validation_split ratio
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
                        logger.info(f"Cached validation length for {cache_key}: {val_length} (from train: {train_length})")
                        return val_length
                    else:
                        return 0
                else:
                    logger.warning(f"No train split found to compute validation length")
                    return 0
            
            if actual_split_name in non_streaming_dataset:
                length = len(non_streaming_dataset[actual_split_name])
                
                # Apply language filtering for XFUND
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
        except Exception as e:
            logger.warning(f"Could not compute length for {cache_key}: {e}")
            return 0
    
    def get_dataset_length(self, split_name: str) -> int:
        """Get the actual length of a dataset split
        
        Args:
            split_name: Split name (train, test, validation)
            
        Returns:
            Actual dataset length
        """
        return self._compute_dataset_length(self.dataset_name, self.hf_dataset_name, split_name)

    def _create_streaming_validation_split(self, train_dataset: HFDataset, validation_split: float, shuffle: bool = False) -> Tuple[HFDataset, HFDataset]:
        """Create train/validation split for streaming datasets"""
        try:
            # Add shuffling support for streaming datasets
            if shuffle:
                # Add a seed for reproducible shuffling
                seed = self.config.get("data_processing", {}).get("seed", 42)
                logger.info(f"Applying shuffle to streaming dataset with seed {seed}")
                train_dataset = train_dataset.shuffle(seed=seed, buffer_size=10000)
            
            # For streaming datasets, we need to estimate the total size to create proper splits
            total_length = self.get_dataset_length("train")
            if total_length > 0:
                val_size = int(total_length * validation_split)
                train_size = total_length - val_size
                
                # Use take() and skip() methods which work with streaming datasets  
                val_dataset = train_dataset.take(val_size)  # Take first val_size examples for validation
                train_dataset_split = train_dataset.skip(val_size)  # Skip first val_size examples for training
                
                logger.info(f"Streaming split: validation={val_size}, training={train_size}")
            else:
                # Fallback if we can't determine size
                skip_ratio = int(1.0 / validation_split)
                val_dataset = train_dataset.take(skip_ratio)
                train_dataset_split = train_dataset.skip(skip_ratio)
            
            shuffle_msg = " with shuffling" if shuffle else ""
            logger.info(f"Streaming mode: Created train/val split{shuffle_msg}.")
            return train_dataset_split, val_dataset
            
        except Exception as e:
            logger.warning(f"Could not create proper streaming split: {e}. "
                         "Using full training set for both train and validation.")
            # Fallback: use full dataset for both (not ideal but works)
            return train_dataset, train_dataset


class LayoutLMDataset(Dataset):
    """PyTorch Dataset for LayoutLM models"""

    def __init__(
        self,
        examples: List[DocumentExample],
        tokenizer: Union[LayoutLMTokenizerFast, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
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
        return self._process_example(example)
    
    def _process_example(self, example: DocumentExample) -> Dict[str, torch.Tensor]:
        """Process a single DocumentExample into model inputs"""
        # For LayoutLM v1, we need to handle tokenization differently
        if isinstance(self.tokenizer, LayoutLMTokenizerFast):
            # LayoutLM v1 tokenizer - tokenize words individually then concatenate
            all_tokens = []
            all_token_boxes = []
            all_token_labels = []

            # Add [CLS] token
            all_tokens.append(self.tokenizer.cls_token_id)
            all_token_boxes.append([0, 0, 0, 0])
            all_token_labels.append(0)  # O label for [CLS]

            # Process each word
            for word, bbox, label in zip(example.words, example.bboxes, example.labels):
                # Tokenize the word
                word_tokens = self.tokenizer.tokenize(word)
                word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

                # Add tokens for this word
                for i, token_id in enumerate(word_token_ids):
                    all_tokens.append(token_id)

                    # Clamp bbox coordinates to valid range [0, 1000]
                    clamped_bbox = [
                        max(0, min(1000, bbox[0])),
                        max(0, min(1000, bbox[1])),
                        max(0, min(1000, bbox[2])),
                        max(0, min(1000, bbox[3]))
                    ]
                    all_token_boxes.append(clamped_bbox)

                    # Use B- label for first token, I- label for subsequent tokens
                    if i == 0:
                        # First token of word gets the original label
                        token_label = self.label2id.get(label, 0)
                    else:
                        # Subsequent tokens get I- version if it's a B- label
                        if label.startswith('B-'):
                            i_label = 'I-' + label[2:]
                            token_label = self.label2id.get(i_label, 0)
                        else:
                            token_label = self.label2id.get(label, 0)
                    all_token_labels.append(token_label)

            # Add [SEP] token
            all_tokens.append(self.tokenizer.sep_token_id)
            all_token_boxes.append([1000, 1000, 1000, 1000])
            all_token_labels.append(0)  # O label for [SEP]

            # Truncate if too long
            if len(all_tokens) > self.max_seq_length:
                all_tokens = all_tokens[:self.max_seq_length - 1] + [self.tokenizer.sep_token_id]
                all_token_boxes = all_token_boxes[:self.max_seq_length - 1] + [[1000, 1000, 1000, 1000]]
                all_token_labels = all_token_labels[:self.max_seq_length - 1] + [0]

            # Pad to max_seq_length
            while len(all_tokens) < self.max_seq_length:
                all_tokens.append(self.tokenizer.pad_token_id)
                all_token_boxes.append([0, 0, 0, 0])
                all_token_labels.append(-100)  # Ignore label for [PAD]

            # Create attention mask
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in all_tokens]

            # Prepare output
            item = {
                "input_ids": torch.tensor(all_tokens, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "bbox": torch.tensor(all_token_boxes, dtype=torch.long),
                "labels": torch.tensor(all_token_labels, dtype=torch.long)
            }
        else:
            # LayoutLM v2/v3 tokenizer
            if isinstance(self.tokenizer, LayoutLMv3Tokenizer) and self.include_image and example.image is not None:
                # LayoutLMv3 with image support
                # Convert string labels to integers
                integer_labels = [self.label2id.get(label, 0) for label in example.labels]

                encoding = self.tokenizer(
                    example.words,
                    boxes=example.bboxes,
                    word_labels=integer_labels,
                    images=example.image,  # Pass image directly to v3 tokenizer
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                # Prepare output with visual features
                item = {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "bbox": encoding["bbox"].squeeze(),
                    "pixel_values": encoding["pixel_values"].squeeze(),  # Visual features
                    "labels": encoding["labels"].squeeze()
                }
            else:
                # LayoutLM v2/v3 without images OR LayoutLMv2
                # Convert string labels to integers
                integer_labels = [self.label2id.get(label, 0) for label in example.labels]

                encoding = self.tokenizer(
                    example.words,
                    boxes=example.bboxes,
                    word_labels=integer_labels,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                # Prepare output (text + layout only)
                item = {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "bbox": encoding["bbox"].squeeze(),
                    "labels": encoding["labels"].squeeze()
                }

        # Add image if available and required (for v1/v2 or v3 without integrated image processing)
        if (self.include_image and example.image is not None
                and not isinstance(self.tokenizer, LayoutLMv3Tokenizer)):
            # Convert PIL image to tensor for v1/v2
            image_tensor = torch.tensor(np.array(example.image)).permute(2, 0, 1).float() / 255.0
            item["image"] = image_tensor
        return item


class LayoutLMStreamingDataset(IterableDataset):
    """Streaming PyTorch Dataset for LayoutLM models"""

    def __init__(
        self,
        dataset_loader: BaseDatasetLoader,
        dataset: HFDataset,
        tokenizer: Union[LayoutLMTokenizerFast, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer],
        label2id: Dict[str, int],
        max_seq_length: int = 512,
        include_image: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        split_name: str = "train"
    ):
        self.dataset_loader = dataset_loader
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.include_image = include_image
        self.split_name = split_name
        
        # Get actual dataset length
        try:
            self._length = dataset_loader.get_dataset_length(split_name)
            logger.info(f"LayoutLMStreamingDataset created for '{split_name}' split with length: {self._length}")
        except Exception as e:
            logger.warning(f"Could not get dataset length for split '{split_name}': {e}. Using 0.")
            self._length = 0
        
        # Apply shuffling if requested
        if shuffle:
            logger.info(f"Applying shuffle to streaming dataset with seed {seed}")
            self.dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            self.dataset = dataset
    
    def __len__(self) -> int:
        """Return the actual dataset length"""
        return self._length

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset and yield processed examples"""
        for item in self.dataset:
            example = self.dataset_loader._process_single_item(item)
            if example is not None:  # Skip None examples (e.g., from CORD dataset)
                yield self._process_example(example)
    
    def _process_example(self, example: DocumentExample) -> Dict[str, torch.Tensor]:
        """Process a single DocumentExample into model inputs"""
        # For LayoutLM v1, we need to handle tokenization differently
        if isinstance(self.tokenizer, LayoutLMTokenizerFast):
            # LayoutLM v1 tokenizer - tokenize words individually then concatenate
            all_tokens = []
            all_token_boxes = []
            all_token_labels = []

            # Add [CLS] token
            all_tokens.append(self.tokenizer.cls_token_id)
            all_token_boxes.append([0, 0, 0, 0])
            all_token_labels.append(0)  # O label for [CLS]

            # Process each word
            for word, bbox, label in zip(example.words, example.bboxes, example.labels):
                # Tokenize the word
                word_tokens = self.tokenizer.tokenize(word)
                word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

                # Add tokens for this word
                for i, token_id in enumerate(word_token_ids):
                    all_tokens.append(token_id)

                    # Clamp bbox coordinates to valid range [0, 1000]
                    clamped_bbox = [
                        max(0, min(1000, bbox[0])),
                        max(0, min(1000, bbox[1])),
                        max(0, min(1000, bbox[2])),
                        max(0, min(1000, bbox[3]))
                    ]
                    all_token_boxes.append(clamped_bbox)

                    # Use B- label for first token, I- label for subsequent tokens
                    if i == 0:
                        # First token of word gets the original label
                        token_label = self.label2id.get(label, 0)
                    else:
                        # Subsequent tokens get I- version if it's a B- label
                        if label.startswith('B-'):
                            i_label = 'I-' + label[2:]
                            token_label = self.label2id.get(i_label, 0)
                        else:
                            token_label = self.label2id.get(label, 0)
                    all_token_labels.append(token_label)

            # Add [SEP] token
            all_tokens.append(self.tokenizer.sep_token_id)
            all_token_boxes.append([1000, 1000, 1000, 1000])
            all_token_labels.append(0)  # O label for [SEP]

            # Truncate if too long
            if len(all_tokens) > self.max_seq_length:
                all_tokens = all_tokens[:self.max_seq_length - 1] + [self.tokenizer.sep_token_id]
                all_token_boxes = all_token_boxes[:self.max_seq_length - 1] + [[1000, 1000, 1000, 1000]]
                all_token_labels = all_token_labels[:self.max_seq_length - 1] + [0]

            # Pad to max_seq_length
            while len(all_tokens) < self.max_seq_length:
                all_tokens.append(self.tokenizer.pad_token_id)
                all_token_boxes.append([0, 0, 0, 0])
                all_token_labels.append(-100)  # Ignore label for [PAD]

            # Create attention mask
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in all_tokens]

            # Prepare output
            item = {
                "input_ids": torch.tensor(all_tokens, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "bbox": torch.tensor(all_token_boxes, dtype=torch.long),
                "labels": torch.tensor(all_token_labels, dtype=torch.long)
            }
        else:
            # LayoutLM v2/v3 tokenizer
            if isinstance(self.tokenizer, LayoutLMv3Tokenizer) and self.include_image and example.image is not None:
                # LayoutLMv3 with image support
                # Convert string labels to integers
                integer_labels = [self.label2id.get(label, 0) for label in example.labels]

                encoding = self.tokenizer(
                    example.words,
                    boxes=example.bboxes,
                    word_labels=integer_labels,
                    images=example.image,  # Pass image directly to v3 tokenizer
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                # Prepare output with visual features
                item = {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "bbox": encoding["bbox"].squeeze(),
                    "pixel_values": encoding["pixel_values"].squeeze(),  # Visual features
                    "labels": encoding["labels"].squeeze()
                }
            else:
                # LayoutLM v2/v3 without images OR LayoutLMv2
                # Convert string labels to integers
                integer_labels = [self.label2id.get(label, 0) for label in example.labels]

                encoding = self.tokenizer(
                    example.words,
                    boxes=example.bboxes,
                    word_labels=integer_labels,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                # Prepare output (text + layout only)
                item = {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "bbox": encoding["bbox"].squeeze(),
                    "labels": encoding["labels"].squeeze()
                }

        # Add image if available and required (for v1/v2 or v3 without integrated image processing)
        if (self.include_image and example.image is not None
                and not isinstance(self.tokenizer, LayoutLMv3Tokenizer)):
            # Convert PIL image to tensor for v1/v2
            image_tensor = torch.tensor(np.array(example.image)).permute(2, 0, 1).float() / 255.0
            item["image"] = image_tensor

        return item


class FUNSDDatasetLoader(BaseDatasetLoader):
    """DatasetLoader for FUNSD dataset"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hf_dataset_name = config["dataset"]["hf_dataset_name"]
        self.label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        # Create mappings
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load FUNSD dataset from HuggingFace"""
        logger.info(f"Loading FUNSD dataset from {self.hf_dataset_name} (streaming={self.streaming})")

        dataset = load_dataset(self.hf_dataset_name, streaming=self.streaming)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = dataset.get("validation", None)

        # Create validation split if we don't have one and validation_split is configured
        if val_dataset is None:
            validation_split = self.config["data_processing"].get("validation_split", 0.1)
            if isinstance(validation_split, float) and 0 < validation_split < 1:
                if not self.streaming:
                    # For non-streaming datasets, we can use train_test_split
                    train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                    train_dataset = train_val["train"]
                    val_dataset = train_val["test"]
                else:
                    # For streaming datasets, use the helper method
                    shuffle_train = self.config.get("data_processing", {}).get("shuffle_train", False)
                    train_dataset, val_dataset = self._create_streaming_validation_split(
                        train_dataset, validation_split, shuffle=shuffle_train
                    )

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get FUNSD label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert FUNSD dataset to DocumentExample format"""
        if self.streaming:
            # For streaming datasets, this method shouldn't be used
            # Instead, use create_examples_iterator or the streaming dataset directly
            raise ValueError("create_examples() should not be used with streaming datasets. "
                           "Use create_examples_iterator() instead.")
        
        examples = []
        for item in dataset:
            examples.append(self._process_single_item(item))
        return examples
    
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single FUNSD dataset item into DocumentExample format"""
        words = item["words"]
        bboxes = item["bboxes"]
        ner_tags = item["ner_tags"]
        image = item["image"]

        # Convert numeric labels to string labels
        labels = [self.id2label[tag] for tag in ner_tags]

        # Normalize bboxes if required
        if self.normalize_bbox:
            # Always normalize if requested, regardless of whether we'll use the image
            # Use the original image to get dimensions
            orig_image = item["image"]
            if orig_image is not None:
                try:
                    width, height = orig_image.size
                    bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                              for bbox in bboxes]
                except AttributeError:
                    # Image might be a dict or other format in streaming mode
                    logger.warning(f"Image has no size attribute (type: {type(orig_image)}). Using default dimensions.")
                    width, height = 1000, 1000
                    bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                              for bbox in bboxes]

        # Prepare image (only if we actually need it)
        if image is not None and self.config["dataset"]["preprocessing"]["include_image"]:
            image = self.prepare_image(image)
        else:
            # Don't include image if include_image is False
            image = None
        return DocumentExample(words=words, bboxes=bboxes, labels=labels, image=image)


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
        logger.info(f"Loading CORD dataset from {self.hf_dataset_name} (streaming={self.streaming})")

        dataset = load_dataset(self.hf_dataset_name, streaming=self.streaming)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = dataset.get("validation", None)

        # Create validation split if we don't have one and validation_split is configured
        if val_dataset is None:
            validation_split = self.config["data_processing"].get("validation_split", 0.1)
            if isinstance(validation_split, float) and 0 < validation_split < 1:
                if not self.streaming:
                    # For non-streaming datasets, we can use train_test_split
                    train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                    train_dataset = train_val["train"]
                    val_dataset = train_val["test"]
                else:
                    # For streaming datasets, use the helper method
                    shuffle_train = self.config.get("data_processing", {}).get("shuffle_train", False)
                    train_dataset, val_dataset = self._create_streaming_validation_split(
                        train_dataset, validation_split, shuffle=shuffle_train
                    )

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get CORD label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert CORD dataset to DocumentExample format"""
        if self.streaming:
            raise ValueError("create_examples() should not be used with streaming datasets. "
                           "Use create_examples_iterator() instead.")
        
        examples = []
        for item in dataset:
            example = self._process_single_item(item)
            if example is not None:  # Skip empty examples
                examples.append(example)
        return examples
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[DocumentExample]:
        """Process a single CORD dataset item into DocumentExample format"""
        # Parse the ground truth JSON
        ground_truth = json.loads(item["ground_truth"])
        image = item["image"]

        # Extract words, bboxes, and labels from valid_line
        words: List[str] = []
        bboxes: List[List[int]] = []
        labels: List[str] = []
        for line in ground_truth.get("valid_line", []):
            category = line.get("category", "other")

            # Map CORD categories to our label format
            if category.upper() not in [label.replace("B-", "").replace("I-", "") for label in self.label_list]:
                category = "other"

            line_words = line.get("words", [])
            for idx, word_info in enumerate(line_words):
                words.append(word_info["text"])

                # Convert quad to bbox [x0, y0, x1, y1]
                quad = word_info["quad"]
                x0 = min(quad["x1"], quad["x2"], quad["x3"], quad["x4"])
                y0 = min(quad["y1"], quad["y2"], quad["y3"], quad["y4"])
                x1 = max(quad["x1"], quad["x2"], quad["x3"], quad["x4"])
                y1 = max(quad["y1"], quad["y2"], quad["y3"], quad["y4"])
                bboxes.append([x0, y0, x1, y1])

                # Assign B-/I- labels
                if idx == 0:
                    labels.append(f"B-{category.upper()}")
                else:
                    labels.append(f"I-{category.upper()}")

        if not words:
            # If no valid words found, skip this example.
            return None

        if self.normalize_bbox and image is not None:
            # Normalize bboxes if required.
            try:
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]
            except AttributeError:
                # Image might be a dict or other format in streaming mode
                logger.warning(f"Image has no size attribute (type: {type(image)}). Using default dimensions.")
                width, height = 1000, 1000
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]

        # Prepare image (only if we actually need it).
        if image is not None and self.config["dataset"]["preprocessing"]["include_image"]:
            image = self.prepare_image(image)
        else:
            # Don't include image if include_image is False.
            image = None
        return DocumentExample(words=words, bboxes=bboxes, labels=labels, image=image)


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
        logger.info(f"Loading SROIE dataset from {self.hf_dataset_name} (streaming={self.streaming})")

        dataset = load_dataset(self.hf_dataset_name, streaming=self.streaming)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Create validation split if needed
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            if not self.streaming:
                # For non-streaming datasets, we can use train_test_split
                train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = train_val["train"]
                val_dataset = train_val["test"]
            else:
                # For streaming datasets, use the helper method
                train_dataset, val_dataset = self._create_streaming_validation_split(
                    train_dataset, validation_split
                )
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get SROIE label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert SROIE dataset to DocumentExample format"""
        if self.streaming:
            raise ValueError("create_examples() should not be used with streaming datasets. "
                           "Use create_examples_iterator() instead.")
        
        examples = []
        for item in dataset:
            examples.append(self._process_single_item(item))
        return examples
    
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single SROIE dataset item into DocumentExample format"""
        words = item["words"]

        # Handle different SROIE dataset formats
        if "bboxes" in item:
            # darentang/sroie format
            bboxes = item["bboxes"]
            ner_tags = item["ner_tags"]
            # For darentang/sroie, we need to load image from path
            if "image_path" in item:
                # For now, we'll skip image loading and set to None
                # since LayoutLMv3 text-only mode doesn't need images
                image = None
            else:
                image = item.get("image", None)
        elif "actual_boxes" in item:
            # buthaya/sroie format
            bboxes = item["actual_boxes"]
            labels_str = item["labels"]
            # Convert string labels to numeric then back to our format
            ner_tags = [self.label2id.get(label, 0) for label in labels_str]
            image = item.get("image", None)
        else:
            raise ValueError(f"Unknown SROIE dataset format. Available keys: {list(item.keys())}")

        # Convert numeric labels to string labels
        labels = [self.id2label[tag] for tag in ner_tags]

        # Normalize bboxes if required
        if self.normalize_bbox:
            if image is not None:
                # Use actual image dimensions
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]
            else:
                # For cases without image, use default normalization
                # Check if we have page dimensions from the dataset
                if "page_width" in item and "page_height" in item:
                    width, height = item["page_width"], item["page_height"]
                else:
                    # Use reasonable defaults for SROIE documents
                    width, height = 1000, 1000
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]

        # Prepare image (only if we actually need it)
        if image is not None and self.config["dataset"]["preprocessing"]["include_image"]:
            image = self.prepare_image(image)
        else:
            # Don't include image if include_image is False
            image = None
        return DocumentExample(words=words, bboxes=bboxes, labels=labels, image=image)


class XFUNDDatasetLoader(BaseDatasetLoader):
    """Loader for XFUND dataset (multilingual FUNSD)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # XFUND uses same labels as FUNSD but supports multiple languages
        self.label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # Language code for XFUND (e.g., "xfund.zh", "xfund.ja", etc.)
        self.language = config["dataset"].get("language", "zh")

    def load_data(self) -> Tuple[HFDataset, HFDataset, Optional[HFDataset]]:
        """Load XFUND dataset from HuggingFace and filter by language"""
        logger.info(f"Loading XFUND dataset from {self.hf_dataset_name} and filtering by language: {self.language} "
                    f"(streaming={self.streaming})")
        dataset = load_dataset(self.hf_dataset_name, streaming=self.streaming)

        # Create validation split from training data
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            if not self.streaming:
                # For non-streaming datasets, we can use train_test_split
                train_val = dataset["train"].train_test_split(test_size=validation_split, seed=42)
                train_dataset = train_val["train"]
                val_dataset = train_val["test"]
            else:
                # For streaming datasets, use the helper method
                shuffle_train = self.config.get("data_processing", {}).get("shuffle_train", False)
                train_dataset, val_dataset = self._create_streaming_validation_split(
                    dataset["train"], validation_split, shuffle=shuffle_train
                )
        else:
            train_dataset = dataset["train"]
            val_dataset = None

        # Filter by language for each split.
        train_dataset = train_dataset.filter(lambda example: example["id"].startswith(f"{self.language}_"))
        val_dataset = val_dataset.filter(lambda example: example["id"].startswith(f"{self.language}_")) if val_dataset else None
        test_dataset = dataset["val"].filter(lambda example: example["id"].startswith(f"{self.language}_"))
        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get XFUND label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert XFUND dataset to DocumentExample format"""
        if self.streaming:
            raise ValueError("create_examples() should not be used with streaming datasets. Use "
                             "create_examples_iterator() instead.")
        
        examples = []
        for item in dataset:
            examples.append(self._process_single_item(item))
        return examples
    
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single XFUND dataset item into DocumentExample format"""
        words = item["words"]
        bboxes = item["bboxes"]
        ner_tags = item["ner_tags"]
        image = item["image"]

        # Convert numeric labels to string labels
        labels = [self.id2label[tag] for tag in ner_tags]

        # Normalize bboxes if required
        if self.normalize_bbox:
            if image is not None:
                if isinstance(image, dict):
                    image = Image.open(io.BytesIO(image["bytes"]))
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]
            else:
                # For cases without image, use default normalization.
                width, height = 1000, 1000  # Default page size for XFUND.
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height) for bbox in bboxes]

        # Prepare image (only if we actually need it)
        if image is not None and self.config["dataset"]["preprocessing"]["include_image"]:
            image = self.prepare_image(image)
        else:
            # Don't include image if include_image is False
            image = None
        return DocumentExample(words=words, bboxes=bboxes, labels=labels, image=image)


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
        logger.info(f"Loading WildReceipt dataset from {self.hf_dataset_name} (streaming={self.streaming})")

        dataset = load_dataset(self.hf_dataset_name, streaming=self.streaming)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Create validation split if needed
        validation_split = self.config["data_processing"].get("validation_split", 0.1)
        if isinstance(validation_split, float) and 0 < validation_split < 1:
            if not self.streaming:
                # For non-streaming datasets, we can use train_test_split
                train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = train_val["train"]
                val_dataset = train_val["test"]
            else:
                # For streaming datasets, use the helper method
                train_dataset, val_dataset = self._create_streaming_validation_split(
                    train_dataset, validation_split
                )
        else:
            val_dataset = None

        return train_dataset, test_dataset, val_dataset

    def get_label_list(self) -> List[str]:
        """Get WildReceipt label list"""
        return self.label_list

    def create_examples(self, dataset: HFDataset) -> List[DocumentExample]:
        """Convert WildReceipt dataset to DocumentExample format"""
        if self.streaming:
            raise ValueError("create_examples() should not be used with streaming datasets. "
                           "Use create_examples_iterator() instead.")
        
        examples = []
        for item in dataset:
            examples.append(self._process_single_item(item))
        return examples
    
    def _process_single_item(self, item: Dict[str, Any]) -> DocumentExample:
        """Process a single WildReceipt dataset item into DocumentExample format"""
        words = item["words"]
        bboxes = item["bboxes"]
        ner_tags = item["ner_tags"]
        image = item["image"]

        # Convert numeric labels to string labels
        labels = [self.id2label[tag] for tag in ner_tags]

        # Normalize bboxes if required
        if self.normalize_bbox:
            if image is not None:
                # Use actual image dimensions
                width, height = image.size
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]
            else:
                # For cases without image, use default normalization
                width, height = 1000, 1000  # Default page size for WildReceipt
                bboxes = [self.normalize_bbox_coordinates(bbox, width, height)
                          for bbox in bboxes]

        # Prepare image (only if we actually need it)
        if image is not None and self.config["dataset"]["preprocessing"]["include_image"]:
            image = self.prepare_image(image)
        else:
            # Don't include image if include_image is False
            image = None
        return DocumentExample(words=words, bboxes=bboxes, labels=labels, image=image)


def create_data_loader(
    examples: Optional[List[DocumentExample]] = None,
    tokenizer: Optional[Union[LayoutLMTokenizerFast, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer]] = None,
    label2id: Optional[Dict[str, int]] = None,
    config: Optional[Dict[str, Any]] = None,
    is_training: bool = True,
    dataset_loader: Optional[BaseDatasetLoader] = None,
    hf_dataset: Optional[HFDataset] = None,
    split_name: str = "train"
) -> DataLoader:
    """Create DataLoader for LayoutLM training/evaluation
    
    Args:
        examples: List of DocumentExample objects (for non-streaming mode)
        tokenizer: Tokenizer for processing text
        label2id: Mapping from label strings to integers
        config: Configuration dictionary
        is_training: Whether this is for training
        dataset_loader: Dataset loader instance (for streaming mode)
        hf_dataset: HuggingFace dataset (for streaming mode)
    """
    
    # Check if we're in streaming mode
    if config and config["dataset"].get("streaming", False):
        if dataset_loader is None or hf_dataset is None:
            raise ValueError("dataset_loader and hf_dataset must be provided for streaming mode")
        
        # Use streaming dataset
        shuffle_train = is_training and config.get("data_processing", {}).get("shuffle_train", False)
        seed = config.get("data_processing", {}).get("seed", 42)
        
        dataset = LayoutLMStreamingDataset(
            dataset_loader=dataset_loader,
            dataset=hf_dataset,
            tokenizer=tokenizer,
            label2id=label2id,
            max_seq_length=config["dataset"]["preprocessing"]["max_seq_length"],
            include_image=config["dataset"]["preprocessing"]["include_image"],
            shuffle=shuffle_train,
            seed=seed,
            split_name=split_name
        )
        
        # For streaming datasets, length is unknown but shuffling is supported
        # Use fewer workers for streaming to avoid worker issues
        return DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=0,  # Use 0 workers for streaming datasets to avoid multiprocessing issues
            pin_memory=True
        )
    else:
        # Use regular dataset
        if examples is None:
            raise ValueError("examples must be provided for non-streaming mode")
        
        dataset = LayoutLMDataset(
            examples=examples,
            tokenizer=tokenizer,
            label2id=label2id,
            max_seq_length=config["dataset"]["preprocessing"]["max_seq_length"],
            include_image=config["dataset"]["preprocessing"]["include_image"]
        )

        return DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=is_training and config["data_processing"]["shuffle_train"],
            num_workers=4,
            pin_memory=True
        )


# Dataset factory
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
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Supported datasets: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[dataset_name](config)


def safe_dataset_length(dataset: Union[HFDataset, LayoutLMStreamingDataset], streaming: bool = False, dataset_loader: Optional[BaseDatasetLoader] = None, split_name: str = "train") -> Union[int, str]:
    """Safely get dataset length, handling streaming mode
    
    Args:
        dataset: HuggingFace dataset or LayoutLMStreamingDataset
        streaming: Whether the dataset is in streaming mode
        dataset_loader: Optional dataset loader to get actual length
        split_name: Split name for computing length
        
    Returns:
        Dataset length as int, or descriptive string if unavailable
    """
    # If it's a LayoutLMStreamingDataset, it has a __len__ method
    if isinstance(dataset, LayoutLMStreamingDataset):
        return len(dataset)
    
    # If we have a dataset_loader and we're in streaming mode, get actual length
    if streaming and dataset_loader is not None:
        try:
            actual_length = dataset_loader.get_dataset_length(split_name)
            if actual_length > 0:
                return actual_length
        except Exception as e:
            logger.warning(f"Could not get actual dataset length: {e}")
    
    # Fallback for non-streaming mode
    if not streaming:
        try:
            return len(dataset)
        except Exception:
            return "Unknown (length unavailable)"
    
    return "Unknown (streaming mode)"
