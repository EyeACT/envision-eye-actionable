"""ONNX Runtime wrapper for the 7-class eye-modality classifier.

The survey never imports torch: the checkpoint is exported once with
``envision-survey export-onnx`` on a machine that has torch + timm, and the
.onnx file is all the VM needs (onnxruntime is CPU-only and light).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .constants import CLASSES

logger = logging.getLogger(__name__)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sidecar(model_path: Path) -> dict:
    """The model's ``.json`` sidecar written by export-onnx (class order and
    provenance), checked against the model file.

    Raises ValueError when the sidecar is unreadable, its ``onnx_sha256``
    differs from the file's hash (a stale sidecar or a swapped model would
    put the wrong checkpoint into the workbook), or its class order differs
    from CLASSES. Warns when there is no sidecar or no parity result.
    Returns {} without a sidecar."""
    model_path = Path(model_path)
    side = model_path.with_suffix(".json")
    meta: dict = {}
    if side.exists():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"model sidecar {side} is unreadable: {e}") from e
        want = meta.get("onnx_sha256")
        if want:
            got = _sha256(model_path)
            if got != want:
                raise ValueError(f"{model_path.name} sha256 {got[:12]}... does not match its sidecar "
                                 f"onnx_sha256 {str(want)[:12]}... (stale sidecar or swapped model)")
        if meta.get("parity_max_abs_diff") is None:
            logger.warning("model sidecar %s has no parity result: this export was never checked "
                           "against PyTorch", side)
    else:
        logger.warning("no sidecar %s next to the model: class order is assumed and the workbook will "
                       "carry no checkpoint provenance", side)
    classes = list(meta.get("classes") or CLASSES)
    if classes != CLASSES:
        raise ValueError(f"model class order {classes} != expected {CLASSES}")
    return meta


class OnnxClassifier:
    """Batch inference with softmax output in ``CLASSES`` order."""

    def __init__(self, model_path: Path, threads: int = 2):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.meta: dict = read_sidecar(self.model_path)
        self.classes = list(CLASSES)
        ort.set_default_logger_severity(3)  # hide harmless device-discovery warnings on VMs
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(N, 3, 224, 224) float32 -> (N, 7) softmax probabilities."""
        logits = self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)
