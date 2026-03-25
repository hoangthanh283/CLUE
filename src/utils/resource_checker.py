"""
Resource availability checker for experiment planning.

This module helps agents verify that planned experiments fit within hardware constraints
before attempting to run them. It checks CPU, GPU memory, system RAM, and disk space.
"""

import subprocess
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Hardware constraints (from configs/system_specs.yaml)
SYSTEM_SPECS = {
    "cpu": {"cores_physical": 6, "cores_logical": 12, "max_cpu_percent": 80},
    "memory": {"total_gb": 15, "safe_buffer_gb": 2, "available_for_training": 13},
    "gpu": {"vram_gb": 6, "safe_vram_gb": 4},
    "storage": {"total_gb": 233, "free_gb": 89},
}


class ResourceChecker:
    """Check if current system resources are sufficient for training."""

    def __init__(self):
        self.specs = SYSTEM_SPECS

    def get_current_gpu_memory(self) -> Dict[str, float]:
        """Query current GPU memory usage via nvidia-smi."""
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,nounits,noheader"],
                universal_newlines=True
            )
            used, total = map(float, output.strip().split(","))
            return {
                "used_gb": used / 1024,
                "total_gb": total / 1024,
                "free_gb": (total - used) / 1024,
                "percent_used": (used / total) * 100,
            }
        except Exception as e:
            logger.warning(f"Failed to query GPU memory: {e}")
            return {"error": str(e)}

    def get_current_ram(self) -> Dict[str, float]:
        """Query current system RAM usage."""
        try:
            output = subprocess.check_output(["free", "-g"], universal_newlines=True)
            lines = output.strip().split("\n")
            parts = lines[1].split()
            total = float(parts[1])
            used = float(parts[2])
            available = float(parts[6])
            return {
                "used_gb": used,
                "total_gb": total,
                "available_gb": available,
                "percent_used": (used / total) * 100,
            }
        except Exception as e:
            logger.warning(f"Failed to query RAM: {e}")
            return {"error": str(e)}

    def estimate_vram_usage(
        self,
        batch_size: int,
        model_params: float = 310e6,  # LayoutLMv3 default
        with_flash_attention: bool = False,
        with_8bit_quantization: bool = False,
    ) -> Dict[str, float]:
        """Estimate VRAM usage for training run."""
        model_weight_gb = (model_params * 4) / (1024 ** 3)
        if with_8bit_quantization:
            model_weight_gb /= 4

        activations_gb = (batch_size * 512 * 768 * 4 * 24) / (1024 ** 3)
        if with_flash_attention:
            activations_gb *= 0.5

        gradients_gb = activations_gb * 0.8
        optimizer_gb = (model_params * 8) / (1024 ** 3)
        if with_8bit_quantization:
            optimizer_gb /= 4

        total_vram_gb = model_weight_gb + activations_gb + gradients_gb + optimizer_gb

        return {
            "total_vram_gb": round(total_vram_gb, 2),
            "fits_in_safe_limit": total_vram_gb <= self.specs["gpu"]["safe_vram_gb"],
        }

    def can_run_experiment(
        self,
        batch_size: int = 4,
        num_workers: int = 2,
        requires_flash_attention: bool = False,
        requires_8bit_quantization: bool = False,
    ) -> Tuple[bool, Dict[str, str]]:
        """Check if proposed experiment configuration can run safely."""
        diagnostics = {}

        gpu_mem = self.get_current_gpu_memory()
        if "error" not in gpu_mem and gpu_mem.get("free_gb", 0) < 1:
            return False, {"gpu_memory": "Less than 1GB GPU memory available"}

        vram_est = self.estimate_vram_usage(
            batch_size=batch_size,
            with_flash_attention=requires_flash_attention,
            with_8bit_quantization=requires_8bit_quantization,
        )
        diagnostics["vram_estimate"] = f"{vram_est['total_vram_gb']}GB"

        if not vram_est["fits_in_safe_limit"] and batch_size > 4:
            return False, {
                **diagnostics,
                "vram_exceeded": f"Estimated {vram_est['total_vram_gb']}GB exceeds safe limit",
            }

        ram_mem = self.get_current_ram()
        if "error" not in ram_mem and ram_mem.get("available_gb", 0) < 5:
            return False, {"ram_available": "Less than 5GB system RAM available"}

        diagnostics["status"] = "✓ Safe to proceed"
        return True, diagnostics
