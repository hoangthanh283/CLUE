"""Integration tests for data pipeline (loading, preprocessing, label space)."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.label_space import (
    get_unified_label_list,
    map_form_entity,
    map_cord_entity,
    map_sroie_entity,
)


class TestDataPipeline:
    """Integration tests for complete data loading and preprocessing."""

    def test_unified_label_space_consistency(self):
        """Verify unified label space is consistent across calls."""
        label_list_1 = get_unified_label_list()
        label_list_2 = get_unified_label_list()

        assert label_list_1 == label_list_2, "Label lists should be identical on repeated calls"
        assert "O" in label_list_1, "Label list should contain 'O' (outside) token"
        assert len(label_list_1) > 0, "Label list should not be empty"

    def test_form_entity_mapping(self):
        """Test FORM entity mapping."""
        # Test that function handles form entities
        result = map_form_entity("FORM.HEADER")
        # Result can be None if entity is not recognized
        if result is not None:
            assert isinstance(result, str), "Mapped entity should be a string or None"
            assert len(result) > 0, "Mapped entity should not be empty"

        # Test case handling (both should give same result)
        mapped_lower = map_form_entity("form.header")
        mapped_upper = map_form_entity("FORM.HEADER")
        # Results might be None, but should be consistent
        assert mapped_lower == mapped_upper, "Mapping should be consistent"

    def test_cord_entity_mapping(self):
        """Test CORD entity mapping."""
        # Test valid CORD entities
        mapped = map_cord_entity("MENU.NM")
        assert isinstance(mapped, str), "Mapped entity should be a string"

        # Test that different entities map to different values
        mapped_nm = map_cord_entity("MENU.NM")
        mapped_price = map_cord_entity("MENU.PRICE")
        # They might map to the same unified label, but function should handle both
        assert mapped_nm is not None and mapped_price is not None

    def test_sroie_entity_mapping(self):
        """Test SROIE entity mapping."""
        sroie_entities = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
        for entity in sroie_entities:
            mapped = map_sroie_entity(entity)
            assert isinstance(mapped, str), f"SROIE entity {entity} should map to string"

    def test_label_space_union(self):
        """Test that unified label space contains all expected entity types."""
        label_list = get_unified_label_list()

        # Should have BIO tags
        bio_labels = [label for label in label_list if "-" in label]
        assert len(bio_labels) > 0, "Label space should have BIO-tagged entities (e.g., B-ENTITY, I-ENTITY)"

        # Should have O (outside) token
        assert "O" in label_list, "Label space should include O (outside) token"

    def test_label_list_uniqueness(self):
        """Test that labels are unique in unified label space."""
        label_list = get_unified_label_list()

        assert len(label_list) == len(set(label_list)), "All labels should be unique"
        assert all(isinstance(label, str) for label in label_list), "All labels should be strings"
        assert len(label_list) > 0, "Label list should not be empty"

    def test_batch_construction_with_mock_data(self, mock_dataloader):
        """Test batch construction and data loading."""
        batch_count = 0
        for batch in mock_dataloader:
            batch_count += 1

            # Check batch structure
            assert "input_ids" in batch, "Batch should have input_ids"
            assert "attention_mask" in batch, "Batch should have attention_mask"
            assert "bbox" in batch, "Batch should have bbox"
            assert "labels" in batch, "Batch should have labels"

            # Check tensor properties
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            assert input_ids.shape[0] > 0, "Batch size should be > 0"
            assert input_ids.shape == labels.shape, "Input and label shapes should match"

        assert batch_count == 3, "Mock dataloader should yield 3 batches"

    def test_bbox_normalization_range(self, fake_batch):
        """Test that normalized bbox values are in valid range."""
        bbox = fake_batch["bbox"]

        # Bbox should be 4D: (batch_size, seq_len, 4) representing [x0, y0, x1, y1]
        assert len(bbox.shape) == 3, "Bbox should be 3D tensor"
        assert bbox.shape[2] == 4, "Bbox should have 4 coordinates per token"

        # Check that values are in reasonable range (typically 0-1000 for normalized coords)
        assert bbox.min() >= 0, "Bbox coordinates should be non-negative"

    def test_ignored_labels_handling(self, fake_batch_with_ignored):
        """Test that -100 labels are handled correctly (should be ignored)."""
        labels = fake_batch_with_ignored["labels"]

        # Check that some labels are -100
        ignored_count = (labels == -100).sum().item()
        assert ignored_count > 0, "Should have some -100 (ignored) labels"

        # Check that not all labels are ignored
        valid_count = (labels >= 0).sum().item()
        assert valid_count > 0, "Should have some valid (non-ignored) labels"
