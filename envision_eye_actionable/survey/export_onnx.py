"""Export the timm checkpoint to ONNX (run where torch + timm are installed).

Example (on the GPU box, in a venv with torch, timm, onnx and onnxruntime;
the file also runs standalone, without this package installed):

    python export_onnx.py --checkpoint ckpt/student_regnety400_synthonly_distill_s2.pt \\
        --arch regnety_004 --out regnety004_synthonly_distill_s2.onnx \\
        --parity-dir data/tier3_images

Writes ``<out>.onnx`` plus a ``<out>.json`` sidecar (class order, arch,
checkpoint name and sha256, preprocessing) that the survey reads and copies
into the workbook README. The export is checked against PyTorch on random
input before it is accepted (max |logit diff| < 1e-4 and identical argmax);
onnxruntime is required for that check, and the .onnx is written to a temp
name and moved to ``--out`` only after it passes.
With ``--parity-dir`` (a folder with one sub-folder per class, such as the
tier-3 training images) it is also checked on ``--parity-per-class`` real
images per class, through the training eval transform; the numbers go into
the sidecar under ``parity_real_images``. The same picks are then fed
through the survey's own decoding and preprocessing (``images.load_frame``
and ``images.preprocess``) and recorded as ``parity_survey_preprocess``,
refused on the same rules; it is recorded as skipped only when the survey
module cannot be imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .constants import CLASSES, IMG_SIZE, MEAN, STD, TRAIN_STORED_SIZE
except ImportError:  # run as a standalone file on a box without the package
    CLASSES = ["CFP", "IR", "PSC", "FAF", "OCT", "OCTA", "NEG"]
    IMG_SIZE = 224
    TRAIN_STORED_SIZE = 256
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state_dict(ckpt: Path):
    import torch
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict", "ema", "state_dict_ema"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if hasattr(sd, "state_dict"):  # a pickled nn.Module
        sd = sd.state_dict()
    # EMA / DataParallel wrappers prefix every key.
    for prefix in ("module.", "model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


PARITY_TOL = 1e-4
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _real_parity(model, sess, root: Path, per_class: int, seed: int) -> dict:
    """ONNX vs PyTorch on real images: ``per_class`` seeded picks from each
    class folder under ``root``, through the training eval transform."""
    import random

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    rng = random.Random(seed)
    items = []
    for cls in CLASSES:
        files = sorted(p for p in (root / cls).rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise RuntimeError(f"no images for class {cls} under {root}")
        for p in rng.sample(files, min(per_class, len(files))):
            items.append((cls, p))
    x = torch.stack([tf(Image.open(p).convert("RGB")) for _, p in items])
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    return {
        "n": len(items),
        "per_class": per_class,
        "seed": seed,
        "max_logit_diff": float(np.abs(ref - got).max()),
        "max_prob_diff": float(np.abs(softmax(ref) - softmax(got)).max()),
        "argmax_agree": int((ref.argmax(1) == got.argmax(1)).sum()),
        "argmax_matches_folder": int(sum(CLASSES[int(i)] == c for i, (c, _) in zip(got.argmax(1), items))),
    }


def _survey_images_module():
    """The survey's own decoder and preprocessing (``images.load_frame`` and
    ``images.preprocess``), or None when it cannot be imported.

    Inside the package it is a relative import. Run as a standalone file,
    ``images.py`` and ``constants.py`` next to this file are loaded as a
    private package (they need numpy and Pillow only)."""
    try:
        from . import images as mod  # type: ignore[attr-defined]
        return mod
    except ImportError:
        pass
    import importlib.util
    import sys
    here = Path(__file__).resolve().parent
    if not (here / "images.py").is_file() or not (here / "constants.py").is_file():
        return None
    name = "_envision_survey_preprocess"
    try:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, here / "constants.py",
                                                          submodule_search_locations=[str(here)])
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[name] = pkg          # package whose __init__ is constants.py (harmless)
            spec.loader.exec_module(pkg)
        return importlib.import_module(f"{name}.images")
    except Exception:  # noqa: BLE001 - parity through the survey path is then reported as skipped
        return None


def _survey_parity(model, sess, root: Path, per_class: int, seed: int) -> dict:
    """ONNX vs PyTorch on the same seeded real-image picks as _real_parity,
    fed through the survey's own input path (images.load_frame, then
    images.preprocess: content crop, squash to 256, resize to 224). Both
    models see the identical tensor, so this checks the export on the
    inputs the survey will actually produce."""
    import random

    import numpy as np
    import torch

    mod = _survey_images_module()
    if mod is None:
        return {"skipped": "survey images module not importable"}
    rng = random.Random(seed)
    items = []
    for cls in CLASSES:
        files = sorted(p for p in (root / cls).rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise RuntimeError(f"no images for class {cls} under {root}")
        for p in rng.sample(files, min(per_class, len(files))):
            items.append((cls, p))
    arrays = []
    for _, p in items:
        loaded = mod.load_frame("raster", p.suffix.lower(), None, p)
        if loaded.image is None:
            raise RuntimeError(f"survey loader could not decode {p}: {loaded.error}")
        arrays.append(mod.preprocess(loaded.image))
    x = np.stack(arrays).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    return {
        "n": len(items),
        "per_class": per_class,
        "seed": seed,
        "path": "images.load_frame + images.preprocess",
        "max_logit_diff": float(np.abs(ref - got).max()),
        "argmax_agree": int((ref.argmax(1) == got.argmax(1)).sum()),
        "argmax_matches_folder": int(sum(CLASSES[int(i)] == c for i, (c, _) in zip(got.argmax(1), items))),
    }


def export(checkpoint: Path, arch: str, out: Path, opset: int = 17, parity_dir: Path | None = None,
           parity_per_class: int = 3, seed: int = 42) -> dict:
    # The parity check is not optional: without onnxruntime there is nothing
    # to check the export with, so refuse before loading or exporting anything.
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("the ONNX parity check needs onnxruntime; install it (pip install onnxruntime) "
                           "and rerun") from e
    import numpy as np
    import timm
    import torch

    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
    missing, unexpected = model.load_state_dict(_load_state_dict(checkpoint), strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={real_missing[:5]} unexpected={list(unexpected)[:5]}")
    model.eval()

    ckpt_sha = _sha256(checkpoint)
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Export to a temp name; only a file that passed the parity check is
    # moved to ``out`` (an existing ``out`` and its sidecar stay untouched
    # when the check fails).
    tmp = out.with_name(out.name + ".tmp")
    export_kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    try:
        try:  # torch >= 2.5 defaults to the dynamo exporter; ask for the classic one
            torch.onnx.export(model, dummy, str(tmp), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(tmp), **export_kwargs)

        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        max_abs = float(np.abs(ref - got).max())
        if max_abs >= PARITY_TOL or (ref.argmax(1) != got.argmax(1)).any():
            raise RuntimeError(f"ONNX parity check failed: max |diff| = {max_abs}")
        real = survey = None
        if parity_dir is not None:
            real = _real_parity(model, sess, Path(parity_dir), parity_per_class, seed)
            if real["max_logit_diff"] >= PARITY_TOL or real["argmax_agree"] != real["n"]:
                raise RuntimeError(f"ONNX parity check on real images failed: {real}")
            # Same picks through the survey's own decoding and preprocessing;
            # refused on the same rules. Skipped (and recorded as skipped)
            # only when the survey module cannot be imported.
            survey = _survey_parity(model, sess, Path(parity_dir), parity_per_class, seed)
            if "skipped" not in survey and (survey["max_logit_diff"] >= PARITY_TOL
                                            or survey["argmax_agree"] != survey["n"]):
                raise RuntimeError(f"ONNX parity check through the survey preprocessing failed: {survey}")
        del sess
        onnx_sha = _sha256(tmp)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)

    meta = {
        "classes": CLASSES,
        "arch": arch,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": ckpt_sha,
        "onnx_sha256": onnx_sha,
        "opset": opset,
        "parity_max_abs_diff": max_abs,
        "parity_real_images": real,
        "parity_survey_preprocess": survey,
        "preprocess": {
            "content_crop": "bbox of gray > max(12, 0.08*max); skip if <2% fg or box <16px",
            "squash_to_square": TRAIN_STORED_SIZE,
            "resize": [IMG_SIZE, IMG_SIZE],
            "note": ("content_crop -> resize((256,256), bilinear, aspect ratio not kept) -> "
                     "resize((224,224), bilinear) == training PNG squash + Resize(224) + CenterCrop(224)"),
            "mean": MEAN, "std": STD, "color": "RGB",
        },
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    side = out.with_suffix(".json")
    side_tmp = side.with_name(side.name + ".tmp")
    side_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    side_tmp.replace(side)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="envision-survey export-onnx",
                                 description="Export the timm classifier checkpoint to ONNX")
    add_arguments(ap)
    return run(ap.parse_args(argv))


def add_arguments(ap: argparse.ArgumentParser):
    ap.add_argument("--checkpoint", required=True, type=Path, help="timm state_dict (.pt)")
    ap.add_argument("--arch", default="regnety_004", help="timm model name (default regnety_004)")
    ap.add_argument("--out", required=True, type=Path, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--parity-dir", type=Path, default=None,
                    help="folder with one sub-folder of real images per class; checks ONNX vs torch on them")
    ap.add_argument("--parity-per-class", type=int, default=3, help="real images per class (default 3)")
    ap.add_argument("--seed", type=int, default=42, help="seed for the parity image picks (default 42)")


def run(args) -> int:
    meta = export(args.checkpoint, args.arch, args.out, args.opset, args.parity_dir,
                  args.parity_per_class, args.seed)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
