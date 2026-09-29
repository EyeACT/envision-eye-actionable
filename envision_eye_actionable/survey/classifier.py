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


LOGITS_OUTPUT = "logits"
NEG_SUBTYPE_OUTPUT = "neg_subtype"
DEFAULT_NEG_SUBTYPES = ["other_eye", "noneye", "eye_nonimage"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pick_outputs(names: list[str], meta: dict) -> tuple[str, str | None, list[str]]:
    """Which ONNX outputs to read: the 7-class logits and, when the model has
    one, the NEG sub-type head.

    Models with several outputs (v2: ``logits``, ``neg_subtype``, ``logits9``)
    are read by name, so the survey never depends on output order; a
    single-output model is read at index 0 as before. Returns
    (logits name, sub-type name or None, sub-type labels)."""
    if not names:
        raise ValueError("ONNX model has no outputs")
    logits = LOGITS_OUTPUT if LOGITS_OUTPUT in names else names[0]
    if len(names) > 1 and LOGITS_OUTPUT not in names:
        raise ValueError(f"ONNX model has outputs {names} but none named {LOGITS_OUTPUT!r}; "
                         "cannot tell which one holds the 7 survey classes")
    sub = NEG_SUBTYPE_OUTPUT if NEG_SUBTYPE_OUTPUT in names else None
    labels = list(meta.get("neg_subtypes") or DEFAULT_NEG_SUBTYPES) if sub else []
    return logits, sub, labels


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
        names = [o.name for o in self.session.get_outputs()]
        self.logits_output, self.subtype_output, self.neg_subtypes = pick_outputs(names, self.meta)
        self._run_names = [self.logits_output] + ([self.subtype_output] if self.subtype_output else [])

    def predict_with_subtype(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(N, 3, 224, 224) float32 -> ((N, 7) softmax probabilities, (N, k)
        NEG sub-type probabilities or None when the model has no sub-type head)."""
        outs = self.session.run(self._run_names, {self.input_name: batch.astype(np.float32, copy=False)})
        logits = np.asarray(outs[0])
        if logits.ndim != 2 or logits.shape[1] != len(CLASSES):
            raise ValueError(f"ONNX output {self.logits_output!r} has shape {logits.shape}, "
                             f"expected (N, {len(CLASSES)})")
        sub = np.asarray(outs[1], dtype=np.float64) if self.subtype_output else None
        return _softmax(logits), sub

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(N, 3, 224, 224) float32 -> (N, 7) softmax probabilities."""
        return self.predict_with_subtype(batch)[0]
