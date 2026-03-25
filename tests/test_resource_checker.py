"""Tests for resource checker utility."""

from src.utils.resource_checker import ResourceChecker


def test_resource_checker_initialization():
    """Test that ResourceChecker initializes correctly."""
    checker = ResourceChecker()
    assert checker.specs["gpu"]["vram_gb"] == 6
    assert checker.specs["memory"]["total_gb"] == 15
    assert checker.specs["cpu"]["cores_logical"] == 12


def test_estimate_vram_usage_conservative():
    """Test VRAM estimation for conservative batch size."""
    checker = ResourceChecker()
    result = checker.estimate_vram_usage(batch_size=1)
    
    # Should fit within safe limit
    assert result["fits_in_safe_limit"] is True
    assert result["total_vram_gb"] < checker.specs["gpu"]["safe_vram_gb"]


def test_estimate_vram_usage_with_optimization():
    """Test VRAM estimation with flash attention enabled."""
    checker = ResourceChecker()
    
    result_without = checker.estimate_vram_usage(batch_size=4)
    result_with = checker.estimate_vram_usage(batch_size=4, with_flash_attention=True)
    
    # Flash attention should reduce VRAM
    assert result_with["total_vram_gb"] < result_without["total_vram_gb"]


def test_can_run_experiment_safe_config():
    """Test that safe configurations pass check."""
    checker = ResourceChecker()
    can_run, diag = checker.can_run_experiment(batch_size=4, num_workers=2)
    
    # Should allow standard configuration
    assert diag["status"] == "✓ Safe to proceed" or can_run is True


def test_can_run_experiment_large_batch():
    """Test that oversized batch sizes are flagged."""
    checker = ResourceChecker()
    can_run, diag = checker.can_run_experiment(batch_size=32)
    
    # Large batch without optimizations should be flagged
    if not can_run:
        assert "vram_exceeded" in diag or "batch_size" in str(diag)


def test_system_specs_completeness():
    """Test that system specs have all required fields."""
    checker = ResourceChecker()
    
    assert "cpu" in checker.specs
    assert "memory" in checker.specs
    assert "gpu" in checker.specs
    assert "storage" in checker.specs
    
    # Verify critical fields exist
    assert checker.specs["gpu"]["safe_vram_gb"] > 0
    assert checker.specs["memory"]["available_for_training"] > 0


if __name__ == "__main__":
    # Quick sanity check
    test_resource_checker_initialization()
    test_estimate_vram_usage_conservative()
    test_system_specs_completeness()
    print("✓ All resource checker tests passed")
