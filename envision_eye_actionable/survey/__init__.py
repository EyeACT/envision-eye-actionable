"""
Zenodo eye-modality survey.

Classifies the images of every record in an envision-discovery Zenodo scrape
(no metadata pre-filter) with the ONNX export of the EyeACT 7-class modality
classifier, rolls per-image header facts up to DICOM-aligned attributes, and
writes AI-READI / CMDS dataset_description + dataset_structure_description
JSON per record. An Excel workbook summarises the run.

Entry point: ``envision-survey`` (see ``survey.cli``). Needs the ``survey``
extra (onnxruntime, openpyxl, jsonschema); never needs torch except for the
one-off ``export-onnx`` step.
"""

__all__ = ["cli"]
