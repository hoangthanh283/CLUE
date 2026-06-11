"""Label remapping for class-incremental (CIL) scenarios.

A CIL session's underlying dataset emits ``ner_tags`` in its OWN native label-id
space (e.g. CORD fine = 61 BIO tags, ``O=0``). The model classifier head, however,
is sized and indexed from the task's ``label_set`` (the cumulative ``[O, session-0
labels, session-1 labels, ...]`` layout grown by ``expand_classifier``). Without a
translation layer the native ids do not line up with the head slots — native ``O``
trains the wrong neuron, and native ids beyond the current head size trip a CUDA
device-side assert (``t >= 0 && t < n_classes``).

``CIL_LabelRemapper`` is the class-incremental analogue of
``doccl.data.dil_remapping.DIL_LabelRemapper``: it wraps a dataset and translates
``labels`` element-wise from native ids into the head-index space for that session,
mapping any out-of-session / unknown native id to ``O`` (head index 0). Remapping is
done lazily in ``__getitem__`` (no extra storage) and ``-100`` (the HuggingFace
ignore index) is preserved.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class CIL_LabelRemapper(Dataset):
    """Wrap a CIL-session dataset and remap native label ids to head-index space.

    Args:
        underlying: a Dataset whose items carry a ``labels`` tensor of native ids.
        native_id_to_label: ``dict[int, str]`` from the underlying dataset
            (its full native BIO label space).
        head_label_to_id: ``dict[str, int]`` — the cumulative model-head label map
            in effect when this session is trained (``O`` at index 0, followed by the
            labels of sessions 0..i). Native labels absent from this map (entities
            from later sessions, already masked to ``O`` by the dataset's filter, or
            any unknown) translate to ``O`` (index 0).
    """

    def __init__(
        self,
        underlying: Dataset,
        native_id_to_label: dict[int, str],
        head_label_to_id: dict[str, int],
    ):
        self.underlying = underlying
        self.head_label_to_id = head_label_to_id
        o_id = head_label_to_id["O"]
        self._id_translation: dict[int, int] = {
            native_id: head_label_to_id.get(native_name, o_id)
            for native_id, native_name in native_id_to_label.items()
        }

    def __len__(self) -> int:
        return len(self.underlying)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.underlying[idx]
        labels = item["labels"]
        remapped = labels.clone()
        for native, head in self._id_translation.items():
            remapped[labels == native] = head
        remapped[labels == -100] = -100  # preserve ignore index
        return {**item, "labels": remapped}
