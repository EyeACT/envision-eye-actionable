"""
envision-eye-actionable — conformer for eye imaging datasets.

Turns raw downloaded archives (from envision-discovery or any other source)
into structured directory trees following the AI-READI dataset description
schema (v0.1.0) that HuggingFace loaders and ML training loops can consume.

Flow per record:
    unpack      → extract every archive recursively
    inventory   → walk tree, classify each file by format/modality/readability
    agent       → Gemma 4 E4B proposes a placement recipe from inventory + README + metadata
    materialize → hardlink files into the target directory tree
    validate    → sample-open one file per modality with the right reader

Default inputs:  ./data/downloads/{source}/{source_id}/  (envision-discovery layout)
Default outputs: ./data/actionable/{source}/{source_id}/

Part of the EyeACT project by the FAIR Data Innovations Hub at CalMI2.
"""

from .pipeline import conform_record, conform_source

__version__ = "0.1.0"
__all__ = ["conform_record", "conform_source", "__version__"]
