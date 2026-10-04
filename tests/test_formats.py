"""Format converters added for the Zenodo 30k rescoring (images.py,
formats.py): files that came out unreadable in the v2 rescoring, rebuilt as
small synthetic fixtures (no data files in the repository).

Each test names the failure it covers: XML error pages and git placeholders
saved under image names, lossless / 12-bit JPEG and JPEG XL, SVG sizes in
physical units, pickled NumPy object arrays, MATLAB files holding objects,
structs and cells, Octave save files, ENVI and BART headers, CIFTI, DICOM
without file meta, Photoshop composites, MP4 recordings without an index,
CZI pixel types and line scans, macOS resource forks in archives.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import types
import zipfile

import numpy as np
import pytest
from PIL import Image

from envision_eye_actionable.survey import formats, images
from envision_eye_actionable.survey.constants import (is_junk_name, sniff_kind, volume_companion,
                                                      volume_header_of)
from envision_eye_actionable.survey.images import load_frame, plane_plan
from envision_eye_actionable.survey.runner import NON_PIXEL_REASONS, unread_reason


def _smooth(shape, seed=0, scale=1000.0):
    """A smooth synthetic image (sum of a few 2D sinusoids plus mild
    noise): neighbouring pixels alike, as in any real image."""
    rng = np.random.default_rng(seed)
    h, w = shape[-2], shape[-1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w))
    for _ in range(4):
        fy, fx, ph = rng.uniform(0.5, 4) / h, rng.uniform(0.5, 4) / w, rng.uniform(0, 6.3)
        img += np.sin(2 * np.pi * (fy * y + fx * x) + ph)
    img = (img - img.min()) / (img.max() - img.min()) * scale
    img += rng.normal(0, scale * 0.01, img.shape)
    if len(shape) == 2:
        return img
    return np.stack([np.roll(img, k, axis=1) for k in range(int(np.prod(shape[:-2])))]).reshape(shape)


# ---------------------------------------------------------------------------
# true content of files with image names
# ---------------------------------------------------------------------------
XML_ERROR = (b'<?xml version="1.0" encoding="UTF-8"?>\n<Error><Code>AccessDenied</Code>'
             b'<Message>Access Denied</Message><RequestId>X</RequestId></Error>')
LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 12345\n")
ANNEX_LINK = b"../../.git/annex/objects/Xx/Yy/SHA256E-s1--" + b"b" * 64 + b".bmp/SHA256E-s1--" + b"b" * 64 + b".bmp"


@pytest.mark.parametrize("payload,what", [
    (XML_ERROR, "XML error page"), (LFS_POINTER, "git-lfs pointer"), (ANNEX_LINK, "git-annex link"),
    (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj", "PDF"), (b"<!DOCTYPE html><html><body>404</body></html>", "html page"),
    (b"", "empty"),
])
def test_placeholders_under_image_names_are_wrong_format(payload, what):
    """250 '.jpeg' files of 50 records were S3 AccessDenied XML pages; .png
    and .bmp files were git-lfs pointers, git-annex links or PDFs. They are
    not pixel-bearing: wrong_format, which does not make a record
    images_unreadable."""
    assert formats.sniff_content(payload) == what
    for kind, ext in (("raster", ".jpeg"), ("raster", ".bmp"), ("array", ".npy"), ("volume", ".nii.gz")):
        ld = load_frame(kind, ext, payload, None)
        assert ld.image is None and ld.error.startswith("wrong_format"), (kind, ld.error)
        assert unread_reason(ld.error) == "wrong_format" and "wrong_format" in NON_PIXEL_REASONS


def test_text_formats_are_not_mistaken_for_placeholders():
    assert formats.sniff_content(b"NRRD0004\n# Complete NRRD file format specification") == "nrrd"
    assert formats.sniff_content(b"ObjectType = Image\nNDims = 3\n") == "metaimage"
    assert formats.sniff_content(b"# Created by Octave 6.4.0\n# name: a\n") == "octave text"
    assert formats.sniff_content(b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>') == "svg"
    assert formats.sniff_content(b"\x89HDF\r\n\x1a\n" + b"\0" * 50) == "hdf5"
    assert formats.sniff_content(b"\x1f\x8b\x08\x00") == "gzip stream"
    assert "gzip stream" not in formats.NON_IMAGE_CONTENT      # a .nii.gz is gzip: never a placeholder


def test_svg_under_an_image_name_is_rendered():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80">'
           b'<rect width="60" height="40" fill="red"/></svg>')
    pytest.importorskip("resvg_py")
    ld = load_frame("raster", ".png", svg, None)
    assert ld.image is not None and "SVG" in ld.facts["conversion"], ld.error


# ---------------------------------------------------------------------------
# JPEG variants Pillow cannot open
# ---------------------------------------------------------------------------
def test_lossless_16bit_jpeg_decodes_through_imagecodecs():
    """iView (Elekta) portal images: lossless SOF3, 16-bit, saved as .jpg."""
    imagecodecs = pytest.importorskip("imagecodecs")
    arr = (_smooth((96, 128), scale=60000)).astype(np.uint16)
    data = imagecodecs.ljpeg_encode(arr)
    assert formats.jpeg_frame_info(data)["process"] == "lossless"
    with pytest.raises(Exception):
        Image.open(io.BytesIO(data)).load()                    # what failed in the v2 rescoring
    ld = load_frame("raster", ".jpg", data, None)
    assert ld.image is not None and ld.image.size == (128, 96), ld.error
    assert ld.facts["source_format"] == "JPEG (lossless, 16-bit)" and ld.facts["lossy"] == "00"
    assert ld.facts["conversion"].startswith("imagecodecs")


def test_12bit_jpeg_and_jpeg_xl_decode():
    imagecodecs = pytest.importorskip("imagecodecs")
    arr12 = (_smooth((64, 80), scale=4000)).astype(np.uint16)
    try:
        j12 = imagecodecs.jpeg8_encode(arr12, bitspersample=12)
    except Exception:  # noqa: BLE001 - build without 12-bit support
        j12 = None
    if j12 is not None:
        ld = load_frame("raster", ".jpg", j12, None)
        assert ld.image is not None and ld.facts.get("bits") in (12, 16), ld.error
    rgb = np.dstack([_smooth((64, 80), seed=s, scale=255) for s in range(3)]).astype(np.uint8)
    jxl = imagecodecs.jpegxl_encode(rgb)
    assert formats.sniff_content(jxl) == "jpegxl"
    ld = load_frame("raster", ".jpg", jxl, None)
    assert ld.image is not None and ld.facts["source_format"] == "JPEG XL" and ld.image.size == (80, 64), ld.error


def test_lossless_jpeg_over_the_budget_is_too_large(monkeypatch):
    imagecodecs = pytest.importorskip("imagecodecs")
    data = imagecodecs.ljpeg_encode(_smooth((64, 64), scale=60000).astype(np.uint16))
    monkeypatch.setattr(images, "DECODE_MAX_BYTES", 1000)
    ld = load_frame("raster", ".jpg", data, None)
    assert ld.image is None and ld.error.startswith("too_large")


def _psd_bytes(h=40, w=50) -> bytes:
    """Minimal Photoshop file: RGB, 8-bit, raw composite image data."""
    rgb = np.dstack([_smooth((h, w), seed=s, scale=255) for s in range(3)]).astype(np.uint8)
    head = b"8BPS" + struct.pack(">H6sHIIHH", 1, b"\0" * 6, 3, h, w, 8, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">H", 0)
    body += b"".join(rgb[..., c].tobytes() for c in range(3))
    return head + body


def test_photoshop_composite_is_read():
    """Pillow numbers PSD layers from 1: seeking frame 0 raised EOFError
    ('attempt to seek outside sequence') on every PSD."""
    ld = load_frame("raster", ".psd", _psd_bytes(), None)
    assert ld.image is not None and ld.image.size == (50, 40), ld.error


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------
def test_svg_sizes_in_points_render():
    """resvg-py's default dpi of 0 turned width="504pt" (matplotlib, R)
    into a zero size: 'SVG has an invalid size' on 993 files."""
    pytest.importorskip("resvg_py")
    svg = (b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="504pt" height="252pt" '
           b'viewBox="0 0 504 252"><rect x="10" y="10" width="200" height="100" fill="#3060a0"/>'
           b'<circle cx="400" cy="120" r="60" fill="#c04020"/></svg>')
    ld = load_frame("vector", ".svg", svg, None)
    assert ld.image is not None and ld.image.size == (512, 256), ld.error
    assert not ld.facts.get("blank")
    for unit in ("mm", "in", "cm"):
        ld = load_frame("vector", ".svg", svg.replace(b"pt", unit.encode()), None)
        assert ld.image is not None, (unit, ld.error)


def test_svg_embedded_raster_with_data_uri_parameters():
    b64 = __import__("base64").b64encode(_png_rgb(64, 48)).decode()
    for uri in (f"data:image/png;charset=utf-8;base64,{b64}", f"data:image/x-png;base64,{b64}",
                f" data:image/PNG;base64,{b64}"):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="10" '
               f'height="10"><image xlink:href="{uri}"/></svg>').encode()
        ld = load_frame("vector", ".svg", svg, None)
        assert ld.image is not None and ld.facts["conversion"] == "SVG embedded PNG image", (uri[:30], ld.facts)


def _png_rgb(w, h) -> bytes:
    rgb = np.dstack([_smooth((h, w), seed=s, scale=255) for s in range(3)]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
def test_pickled_object_npy_is_read_with_the_numpy_allow_list(tmp_path):
    """np.save of a dict (object dtype): mmap failed with 'Python objects in
    dtype'. The pickle is read with NumPy globals only."""
    img = _smooth((80, 90)).astype(np.float32)
    np.save(tmp_path / "d.npy", {"meta": np.arange(5), "image": img}, allow_pickle=True)
    ld = load_frame("array", ".npy", None, tmp_path / "d.npy")
    assert ld.image is not None and ld.facts["array_variable"] == "array.image", ld.error
    assert ld.facts["source_format"] == "NPY (pickled object array)"
    np.save(tmp_path / "l.npy", np.array([np.zeros((3, 3)), img], dtype=object), allow_pickle=True)
    ld = load_frame("array", ".npy", None, tmp_path / "l.npy")
    assert ld.image is not None and ld.facts["array_variable"] == "array[1]"


class _Evil:
    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        return (exec, (f"open({self.target!r}, 'w').write('pwned')",))


def test_pickled_object_npy_never_runs_code(tmp_path):
    target = tmp_path / "pwned.txt"
    np.save(tmp_path / "evil.npy", np.array([_Evil(str(target)), 1], dtype=object), allow_pickle=True)
    ld = load_frame("array", ".npy", None, tmp_path / "evil.npy")
    assert ld.image is None and "refused global builtins.exec" in ld.error
    assert not target.exists()
    assert unread_reason(ld.error) == "not_image_shaped"
    # the same object array inside an npz member
    with zipfile.ZipFile(tmp_path / "evil.npz", "w") as zf:
        buf = io.BytesIO()
        np.save(buf, np.array([_Evil(str(target))], dtype=object), allow_pickle=True)
        zf.writestr("x.npy", buf.getvalue())
        buf = io.BytesIO()
        np.save(buf, _smooth((70, 70)).astype(np.float32))
        zf.writestr("img.npy", buf.getvalue())
    ld = load_frame("array", ".npz", None, tmp_path / "evil.npz")
    assert ld.image is not None and ld.facts["array_variable"] == "img" and not target.exists()
    # an object member holding the image (np.savez of a dict entry)
    np.savez(tmp_path / "obj.npz", meta=np.arange(4), d=np.array({"scan": _smooth((70, 80))}, dtype=object))
    ld = load_frame("array", ".npz", None, tmp_path / "obj.npz")
    assert ld.image is not None and ld.facts["array_variable"] == "d.scan", ld.error


def test_complex_and_channel_first_arrays(tmp_path):
    z = (_smooth((80, 100)) * np.exp(1j * _smooth((80, 100), seed=3) / 100)).astype(np.complex64)
    np.save(tmp_path / "c.npy", z)
    ld = load_frame("array", ".npy", None, tmp_path / "c.npy")
    assert ld.image is not None and "magnitude" in ld.facts["conversion"], ld.error
    rgb = np.stack([_smooth((90, 120), seed=s, scale=255) for s in range(3)]).astype(np.uint8)   # (3, H, W)
    np.save(tmp_path / "chw.npy", rgb)
    ld = load_frame("array", ".npy", None, tmp_path / "chw.npy")
    assert ld.image is not None and ld.image.size == (120, 90) and not ld.facts["rgb_is_gray"]


def test_stack_axis_follows_the_image_axes():
    """Of three axes, the stack is the one outside the two image-like axes
    (not simply the shortest): a stack of 1000 patches of 64 x 64 gives a
    patch, not a 1000 x 64 strip."""
    assert plane_plan((1000, 64, 64))[1] == (64, 64)
    assert plane_plan((64, 64, 1000))[1] == (64, 64)
    assert plane_plan((49, 496, 512))[1] == (496, 512)
    assert plane_plan((496, 512, 49))[1] == (496, 512)
    assert plane_plan((3, 200, 300))[1] == (200, 300)
    assert plane_plan((10, 128, 96))[1] == (128, 96)
    assert plane_plan((4096, 64)) is None and plane_plan((40, 40)) is None


def test_tables_signals_and_noise_are_not_images():
    rng = np.random.default_rng(0)
    table = rng.random((500, 120)) * np.logspace(0, 4, 120)          # columns of unrelated scales
    signals = np.cumsum(rng.normal(size=(2000, 100)), axis=0)        # 100 random walks side by side
    noise = rng.random((300, 300))
    for name, a in (("table", table), ("signals", signals), ("noise", noise)):
        ok, ratio = formats.image_like(a)
        assert not ok and ratio >= formats.IMAGE_LIKE_MAX_RATIO, (name, ratio)
    for a in (_smooth((200, 300)), _smooth((128, 128), seed=5) > 500):   # an image and a mask
        ok, ratio = formats.image_like(a)
        assert ok and ratio < 0.5, ratio
    # speckled OCT-like data: a smooth layered structure times speckle of 2 px grain
    y = np.linspace(0, 1, 256)[:, None]
    layers = 0.2 + np.exp(-((y - 0.4) / 0.05) ** 2) + 0.6 * np.exp(-((y - 0.7) / 0.08) ** 2)
    speckle = np.repeat(np.repeat(rng.exponential(1.0, (128, 256)), 2, axis=0), 1, axis=1)
    ok, ratio = formats.image_like(np.log1p(layers * speckle * 100))
    assert ok, ratio


def test_a_rejected_candidate_gives_way_to_the_next(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)
    with h5py.File(tmp_path / "t.h5", "w") as f:
        f["image_table"] = rng.random((400, 300)) * np.logspace(0, 3, 300)   # name hint, but a table
        f["field"] = _smooth((100, 120))
    ld = load_frame("array", ".h5", None, tmp_path / "t.h5")
    assert ld.image is not None and ld.facts["array_variable"] == "field", ld.error
    with h5py.File(tmp_path / "only_table.h5", "w") as f:
        f["data"] = rng.random((400, 300)) * np.logspace(0, 3, 300)
    ld = load_frame("array", ".h5", None, tmp_path / "only_table.h5")
    assert ld.image is None and "no image-like array among 1" in ld.error
    assert unread_reason(ld.error) == "not_image_shaped"


def test_matlab_char_datasets_are_skipped(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "s.mat", "w", userblock_size=512) as f:
        d = f.create_dataset("name", data=(_smooth((100, 100)) % 90 + 32).astype(np.uint16))
        d.attrs["MATLAB_class"] = np.bytes_("char")
    ld = load_frame("array", ".mat", None, tmp_path / "s.mat")
    assert ld.image is None and "among 0" in ld.error


def test_oct_volume_gives_an_en_face_view(tmp_path):
    h5py = pytest.importorskip("h5py")
    vol = np.stack([_smooth((100, 120), seed=k % 3) for k in range(32)]).astype(np.float32)   # (B-scans, depth, width)
    with h5py.File(tmp_path / "v.h5", "w") as f:
        f["oct_volume"] = vol
        f["other_stack"] = vol
    ld = load_frame("array", ".h5", None, tmp_path / "v.h5")
    assert ld.image is not None and ld.image.size == (120, 100), ld.error
    assert [v[0] for v in ld.views] == ["enface"] and ld.views[0][1].size == (120, 32)
    assert "en-face" in ld.views[0][2]["conversion"]
    with h5py.File(tmp_path / "w.h5", "w") as f:
        f["stack"] = vol                                   # no OCT name: no projection
    ld = load_frame("array", ".h5", None, tmp_path / "w.h5")
    assert ld.image is not None and not ld.views


# ---------------------------------------------------------------------------
# MATLAB v5
# ---------------------------------------------------------------------------
def test_mat5_header_reader_matches_whosmat(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    scipy_io.savemat(tmp_path / "a.mat", {"img": _smooth((70, 90)).astype(np.float32), "v": np.arange(7),
                                          "z": np.ones((3, 4)) * 1j, "s": {"x": 1}}, do_compression=True)
    want = sorted((n, tuple(s), c) for n, s, c in scipy_io.whosmat(tmp_path / "a.mat"))
    got = sorted((n, s, c) for n, s, c, _ in formats.mat5_variables(tmp_path / "a.mat"))
    assert got == want
    assert {n: cx for n, _, _, cx in formats.mat5_variables(tmp_path / "a.mat")}["z"] is True
    scipy_io.savemat(tmp_path / "b.mat", {"img": _smooth((70, 90))}, do_compression=False)
    assert [v[:3] for v in formats.mat5_variables(tmp_path / "b.mat")] == [("img", (70, 90), "double")]


def test_mat5_when_whosmat_fails_on_a_matlab_object(tmp_path, monkeypatch):
    """whosmat raised "'NoneType' object is not iterable" on files that
    hold a MATLAB object (table, string, classdef): the survey's own
    header reader lists the variables instead."""
    scipy_io = pytest.importorskip("scipy.io")
    scipy_io.savemat(tmp_path / "m.mat", {"frame": _smooth((80, 80)).astype(np.float32), "n": np.arange(3)})

    def broken(*a, **k):
        raise TypeError("'NoneType' object is not iterable")
    monkeypatch.setattr(scipy_io, "whosmat", broken)
    ld = load_frame("array", ".mat", None, tmp_path / "m.mat")
    assert ld.image is not None and ld.facts["array_variable"] == "frame", ld.error
    assert "own header reader" in ld.facts["mat_lister"]


def test_mat5_struct_and_cell_contents_are_searched(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    cell = np.empty((1, 2), dtype=object)
    cell[0, 0], cell[0, 1] = "label", _smooth((64, 96))
    scipy_io.savemat(tmp_path / "s.mat", {"data": {"fs": 100.0, "scan": _smooth((90, 110)), "name": "x"},
                                          "c": cell, "vec": np.arange(10)})
    ld = load_frame("array", ".mat", None, tmp_path / "s.mat")
    assert ld.image is not None and ld.facts["array_variable"] == "data.scan", ld.error
    assert "struct" in ld.facts["conversion"]


# ---------------------------------------------------------------------------
# Octave
# ---------------------------------------------------------------------------
def _oct_record(name: str, typ: str, body: bytes) -> bytes:
    n, t = name.encode(), typ.encode()
    return struct.pack("<i", len(n)) + n + struct.pack("<i", 0) + b"\0" + b"\xff" + struct.pack("<i", len(t)) + t + body


def _oct_matrix(a: np.ndarray) -> bytes:
    a = np.asarray(a, "<f8")
    return struct.pack("<i", -a.ndim) + b"".join(struct.pack("<i", d) for d in a.shape) + b"\x07" + \
        a.tobytes(order="F")


def test_octave_binary_save_file(tmp_path):
    """GNU Octave binary saves (Octave-1-L), here named .oct: oct-converter
    took them for Bioptigen OCT and failed with FileNotFoundError."""
    img = _smooth((70, 90))
    body = b"Octave-1-L\x00"
    body += _oct_record("fs", "scalar", b"\x07" + struct.pack("<d", 2.5))
    body += _oct_record("label", "string", struct.pack("<iii", -2, 1, 3) + b"abc")
    fields = _oct_record("vec", "matrix", _oct_matrix(np.arange(5.0)[None])) + \
        _oct_record("bscan", "matrix", _oct_matrix(img))
    body += _oct_record("dataset", "scalar struct", struct.pack("<i", 2) + fields)
    (tmp_path / "d.oct").write_bytes(body)
    got = formats.octave_variables(tmp_path / "d.oct")
    assert [(v[0], v[1]) for v in got] == [("dataset.vec", (1, 5)), ("dataset.bscan", (70, 90))]
    ld = load_frame("vendor_oct", ".oct", None, tmp_path / "d.oct")
    assert ld.image is not None and ld.facts["array_variable"] == "dataset.bscan", ld.error
    assert ld.facts["source_format"].startswith("Octave binary")
    assert np.allclose(np.asarray(formats.octave_array(tmp_path / "d.oct", got[1])), img)


def test_octave_text_save_file(tmp_path):
    img = _smooth((80, 70))
    lines = ["# Created by Octave 6.4.0", "# name: n", "# type: scalar", "3", "", "",
             "# name: frame", "# type: matrix", "# rows: 80", "# columns: 70"]
    lines += [" " + " ".join(f"{v:.4f}" for v in row) for row in img]
    (tmp_path / "g.mat").write_text("\n".join(lines) + "\n")
    pairs = formats.octave_text_variables(tmp_path / "g.mat")
    assert [(n, a.shape) for n, a in pairs] == [("frame", (80, 70))] and np.allclose(pairs[0][1], img, atol=1e-3)
    ld = load_frame("array", ".mat", None, tmp_path / "g.mat")
    assert ld.image is not None and ld.facts["array_variable"] == "frame", ld.error
    assert ld.facts["source_format"] == "Octave text (.mat)"


# ---------------------------------------------------------------------------
# .hdr: ENVI, BART, Analyze; CIFTI
# ---------------------------------------------------------------------------
def _envi(tmp_path, name, lines, samples, bands, interleave="bsq", default=None, data_ext=".img"):
    cube = np.stack([_smooth((lines, samples), seed=b) for b in range(bands)], axis=-1).astype("<f4")  # L,S,B
    order = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    (tmp_path / f"{name}{data_ext}").write_bytes(np.ascontiguousarray(cube.transpose(order)).tobytes())
    hdr = (f"ENVI\ndescription = {{test}}\nsamples = {samples}\nlines = {lines}\nbands = {bands}\n"
           f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = {interleave}\nbyte order = 0\n")
    if default:
        hdr += "default bands = {" + ", ".join(str(b) for b in default) + "}\n"
    (tmp_path / f"{name}.hdr").write_text(hdr)
    return tmp_path / f"{name}.hdr", cube


@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
def test_envi_cube_middle_band(tmp_path, interleave):
    """ENVI headers (.hdr + .img): nibabel took them for Analyze and failed
    ('Cannot work out file type')."""
    hdr, cube = _envi(tmp_path, "c", 80, 96, 7, interleave)
    ld = load_frame("volume_pair", ".hdr", None, hdr)
    assert ld.image is not None and ld.image.size == (96, 80), ld.error
    assert ld.facts["source_format"] == f"ENVI ({interleave})" and "band 4/7" in ld.facts["conversion"]
    got = formats.read_envi(hdr, tmp_path / "c.img")[0]
    assert np.array_equal(np.asarray(got), cube)


def test_envi_default_bands_rgb_and_line_scans(tmp_path):
    hdr, _ = _envi(tmp_path, "rgb", 80, 96, 6, default=(5, 3, 1), data_ext="")
    ld = load_frame("volume_pair", ".hdr", None, hdr)
    assert ld.image is not None and "default bands [5, 3, 1]" in ld.facts["conversion"], ld.error
    hdr, _ = _envi(tmp_path, "line", 1, 640, 20, "bip")
    ld = load_frame("volume_pair", ".hdr", None, hdr)
    assert ld.image is None and "not image shaped: ENVI 1 lines" in ld.error


def test_bart_cfl_magnitude(tmp_path):
    """BART (.hdr '# Dimensions' + .cfl complex64): the data file was never
    fetched next to its header (companion_missing)."""
    img = _smooth((90, 80))
    data = np.zeros((90, 80, 1, 4), np.complex64, order="F")
    for c in range(4):
        data[:, :, 0, c] = img * np.exp(1j * c)
    (tmp_path / "k.cfl").write_bytes(data.tobytes(order="F"))
    (tmp_path / "k.hdr").write_text("# Dimensions\n90 80 1 4 1 1 1 1 1 1 1 1 1 1 1 1\n")
    ld = load_frame("volume_pair", ".hdr", None, tmp_path / "k.hdr")
    assert ld.image is not None and ld.image.size == (80, 90) and "magnitude" in ld.facts["conversion"], ld.error
    (tmp_path / "n.cfl").write_bytes(np.zeros((640, 16), np.complex64).tobytes())
    (tmp_path / "n.hdr").write_text("# Dimensions\n640 1 1 16 1 1 1 1 1 1 1 1 1 1 1 1\n")
    ld = load_frame("volume_pair", ".hdr", None, tmp_path / "n.hdr")
    assert ld.image is None and "not image shaped" in ld.error


def test_header_data_companions():
    assert volume_companion("x/k.hdr", ["x/k.hdr", "x/k.cfl"]) == "x/k.cfl"
    assert volume_companion("c.hdr", ["c.hdr", "c"]) == "c"
    assert volume_companion("c.hdr", ["c.hdr", "c.dat"]) == "c.dat"
    assert volume_companion("a.hdr", ["a.hdr", "a.img", "a.cfl"]) == "a.img"
    assert volume_companion("v.mhd", ["v.mhd", "v.raw"]) == "v.raw"
    assert volume_header_of("k.cfl", ["k.hdr", "k.cfl"]) == "k.hdr"
    assert volume_header_of("c", ["c.hdr", "c"]) == "c.hdr"
    assert volume_header_of("c.dat", ["c.hdr"]) == "c.hdr"
    assert volume_header_of("notes.dat", ["other.hdr"]) is None


def test_top_level_bart_and_envi_data_files_are_fetched(tmp_path):
    from envision_eye_actionable.survey.runner import plan_files
    files = [{"key": k, "size": 10, "md5": None, "url": None}
             for k in ("k.hdr", "k.cfl", "e.hdr", "e", "c.hdr", "c.dat", "notes.dat", "readme.txt")]
    _avail, _zips, to_fetch, skipped = plan_files(files, tmp_path, remote_zip=False)
    assert sorted(f["key"] for f in to_fetch) == ["c.dat", "c.hdr", "e", "e.hdr", "k.cfl", "k.hdr"] and skipped == 2


def test_cifti_is_not_image_shaped(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")

    class FakeCifti:
        shape = (1, 91282)
        header = object()                       # a Cifti2Header has no get_zooms
    monkeypatch.setattr(nib, "load", lambda p: FakeCifti())
    (tmp_path / "p.dlabel.nii").write_bytes(b"\x1c\x02\x00\x00n+2\x00" + b"\0" * 600)
    ld = load_frame("volume", ".nii", None, tmp_path / "p.dlabel.nii")
    assert ld.image is None and "CIFTI-2" in ld.error and unread_reason(ld.error) == "not_image_shaped"


def test_gdal_nodata_raster_is_blank(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "n.tif", np.full((39, 59), -3.4028234663852886e38, np.float32))
    ld = load_frame("raster", ".tif", None, tmp_path / "n.tif")
    assert ld.image is None and ld.error.startswith("blank") and unread_reason(ld.error) == "blank"
    stack = np.full((5, 80, 90), np.nan, np.float32)
    stack[2] = _smooth((80, 90))
    tifffile.imwrite(tmp_path / "bands.tif", stack, planarconfig="separate")
    ld = load_frame("raster", ".tif", None, tmp_path / "bands.tif")
    assert ld.image is not None, ld.error                     # the first band with values


# ---------------------------------------------------------------------------
# DICOM without file meta
# ---------------------------------------------------------------------------
def test_dicom_without_file_meta_decodes(tmp_path):
    """RT dose grids written without file meta: pydicom 3 refused to decode
    ('transfer_syntax_uid' is required)."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    ds.Modality = "RTDOSE"
    ds.Rows, ds.Columns, ds.NumberOfFrames = 64, 80, 5
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.stack([_smooth((64, 80), seed=k, scale=60000) for k in range(5)]).astype("<u2").tobytes()
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, implicit_vr=True, little_endian=True, enforce_file_format=False)
    raw = buf.getvalue()
    assert raw[:2] == b"\x08\x00" and sniff_kind(raw[:132]) == ("dicom", ".dcm")
    ld = load_frame("dicom", ".dcm", raw, None)
    assert ld.image is not None and ld.image.size == (80, 64), ld.error
    assert ld.facts["source_format"] == "DICOM (no file meta, Implicit VR Little Endian)"
    assert "frame 2/5" in ld.facts["conversion"]


def test_bare_dicom_sniff_needs_two_consistent_elements():
    """SQLite pages starting 08 00 were taken for preamble-less DICOM and
    failed to decode (BytesLengthException)."""
    assert sniff_kind(b"\x08\x00\x10\x00" + b"\x00" * 4 + b"\x00\x00\x00\x00" + b"\x03" * 120) is None
    assert sniff_kind(b"\x08\x00\x05\x00\x0a\x00\x00\x00ISO_IR 100" + b"\x08\x00\x08\x00\x04\x00\x00\x00"
                      b"ORIG" + b"\0" * 100) == ("dicom", ".dcm")
    assert sniff_kind(b"\x08\x00\x08\x00CS\x04\x00ORIG\x08\x00\x16\x00UI\x02\x001\x00" + b"\0" * 100) == \
        ("dicom", ".dcm")
    assert sniff_kind(b"\x08\x00\x08\x00CS\x04\x00ORIG\x08\x00\x06\x00UI\x02\x001\x00" + b"\0" * 100) is None
    assert sniff_kind(b"\x08\x00\x05\x00\x00\x10\x00\x00" + b"\x01" * 124) is None       # implicit length 4096
    assert sniff_kind(b"\x08\x00\x08\x00CS\x04\x00ORIG\x09\x00\x10\x00LO\x02\x00ab" + b"\0" * 100) is None


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
def _annexb_h264(n=40, w=64, h=48) -> bytes:
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="h264")
    try:
        s = out.add_stream("libx264", rate=10)
    except Exception:  # noqa: BLE001 - PyAV build without libx264
        pytest.skip("no libx264 encoder")
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    for i in range(n):
        frame = np.full((h, w, 3), min(255, 20 + i * 5), np.uint8)
        frame[: h // 2, : w // 2] = 255 - i * 3
        for pkt in s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            out.mux(pkt)
    for pkt in s.encode():
        out.mux(pkt)
    out.close()
    return buf.getvalue()


def test_mp4_without_index_decodes_as_raw_h264(tmp_path):
    """Recordings that stopped before the moov box was written: the bundled
    ffmpeg 'Could not load meta information'. Their mdat holds an Annex-B
    H.264 stream, which PyAV decodes."""
    stream = _annexb_h264()
    assert stream[:4] == b"\x00\x00\x00\x01"
    mp4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00\x00\x00\x08free" + \
        b"\x00\x00\x00\x00mdat" + stream
    (tmp_path / "rec.mp4").write_bytes(mp4)
    ld = load_frame("video", ".mp4", None, tmp_path / "rec.mp4")
    assert ld.image is not None and ld.image.size == (64, 48), ld.error
    assert "without index" in ld.facts["source_format"] and "PyAV" in ld.facts["conversion"]
    (tmp_path / "junk.mp4").write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 200)
    ld = load_frame("video", ".mp4", None, tmp_path / "junk.mp4")
    assert ld.image is None and "PyAV" in ld.error


# ---------------------------------------------------------------------------
# CZI fallback (czifile), with a stand-in module: czifile cannot write CZI
# ---------------------------------------------------------------------------
class _FakeCziImage:
    def __init__(self, dims, data, start=None, pixeltype=9):
        self.dims, self.data = tuple(dims), data
        self.shape, self.start = data.shape, tuple(start or (0,) * data.ndim)
        self.pixeltype, self.levels, self.nbytes = pixeltype, [self], data.nbytes
        self.selected = None

    def __call__(self, **sel):
        idx = tuple(sel[d] - self.start[i] if d in sel else slice(None) for i, d in enumerate(self.dims))
        sub = _FakeCziImage([d for d in self.dims if d not in sel], self.data[idx], pixeltype=self.pixeltype)
        self.selected = sel
        return sub

    def asarray(self):
        return self.data


def _fake_czifile(monkeypatch, scene):
    mod = types.ModuleType("czifile")

    class CziFile:
        def __init__(self, path):
            self.scenes = [scene]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    mod.CziFile = CziFile
    monkeypatch.setitem(sys.modules, "czifile", mod)
    broken = types.ModuleType("pylibCZIrw")
    monkeypatch.setitem(sys.modules, "pylibCZIrw", broken)        # 'from pylibCZIrw import czi' fails


def test_czi_fallback_bgra_middle_plane(tmp_path, monkeypatch):
    """Bgra32 CZI (pixel type 9): pylibCZIrw raised 'pixel type ... does not
    match'. czifile composes the plane; BGRA becomes RGB."""
    bgra = np.zeros((6, 70, 90, 4), np.uint8)
    bgra[..., 0] = 200                                   # blue
    bgra[..., 2] = (_smooth((70, 90), scale=100)).astype(np.uint8)
    scene = _FakeCziImage("TYXS", bgra, start=(10, 0, 0, 0))
    _fake_czifile(monkeypatch, scene)
    (tmp_path / "a.czi").write_bytes(b"ZISRAWFILE")
    ld = load_frame("microscopy", ".czi", None, tmp_path / "a.czi")
    assert ld.image is not None and ld.image.size == (90, 70), ld.error
    assert scene.selected == {"T": 10}
    r, g, b = np.asarray(ld.image).reshape(-1, 3).mean(axis=0)
    assert b > r                                          # channel order fixed
    assert "czifile" in ld.facts["conversion"]


def test_czi_line_scan_keeps_z_as_rows(tmp_path, monkeypatch):
    scene = _FakeCziImage("TZX", _smooth((5, 120, 200), scale=200).astype(np.uint8), pixeltype=0)
    _fake_czifile(monkeypatch, scene)
    (tmp_path / "xz.czi").write_bytes(b"ZISRAWFILE")
    ld = load_frame("microscopy", ".czi", None, tmp_path / "xz.czi")
    assert ld.image is not None and ld.image.size == (200, 120), ld.error
    assert scene.selected == {"T": 0}


# ---------------------------------------------------------------------------
# archives: macOS resource forks
# ---------------------------------------------------------------------------
def test_macos_resource_forks_are_not_images(tmp_path):
    from envision_eye_actionable.survey.archives import RecordWalker
    assert is_junk_name("__MACOSX/a/._x.png") and is_junk_name("d/._x.tif") and is_junk_name("a\\._b.jpg")
    assert not is_junk_name("data/x.png") and not is_junk_name("data/_x.png")
    zp = tmp_path / "r.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("imgs/a.png", _png_rgb(80, 64))
        zf.writestr("__MACOSX/imgs/._a.png", b"\x00\x05\x16\x07" + b"\0" * 80)
        zf.writestr("imgs/._b.png", b"\x00\x05\x16\x07" + b"\0" * 80)
    w = RecordWalker(scratch=tmp_path / "s")
    w.add_file(zp, zp.name)
    assert [e.member for e in w.entries] == ["imgs/a.png"] and w.kind_counts["junk"] == 2


def test_unread_reason_prefers_the_wrong_format_and_blank_prefixes():
    assert unread_reason("wrong_format: git-annex link saved as .nii (volume data file not available)") == \
        "wrong_format"
    assert unread_reason("blank: no finite pixel value (all NaN or nodata) in (3, 4) (tifffile)") == "blank"
    assert unread_reason("no reader for a .vol without the Heidelberg HSF-OCT signature (unknown content)") == \
        "no_reader"


def test_vol_and_oct_extensions_shared_with_other_formats(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    with mrcfile.new(str(tmp_path / "sub.mrc")) as m:
        m.set_data(_smooth((8, 70, 90)).astype(np.float32))
    (tmp_path / "sub.vol").write_bytes((tmp_path / "sub.mrc").read_bytes())
    ld = load_frame("vendor_oct", ".vol", None, tmp_path / "sub.vol")
    assert ld.image is not None and ld.facts["source_format"] == "MRC volume (.vol)", ld.error
    (tmp_path / "x.vol").write_bytes(b"ep-v0.6.8   2" + bytes(range(256)) * 4)
    ld = load_frame("vendor_oct", ".vol", None, tmp_path / "x.vol")
    assert ld.image is None and unread_reason(ld.error) == "no_reader"
    assert json.dumps(ld.facts)                            # facts stay JSON-serialisable
