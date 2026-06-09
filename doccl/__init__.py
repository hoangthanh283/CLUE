"""DocCL: Continual Learning for Document Understanding.

Master's thesis + AAAI 2027 submission targeting:
    Diagnostic + remedy paper on per-component forgetting in
    multimodal document encoders (LayoutLMv3).

Author: Thanh Hoang (HUST)
"""
from doccl.types import (
    EvalMetrics,
    ModalityMask,
    ScenarioType,
    TaskInfo,
    TaskState,
    TrainMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "EvalMetrics",
    "ModalityMask",
    "ScenarioType",
    "TaskInfo",
    "TaskState",
    "TrainMetrics",
    "__version__",
]
