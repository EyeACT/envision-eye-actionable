"""
envision-eye-actionable — ADDF conformer for eye imaging datasets.

Turns raw downloaded archives (from envision-discovery or any other source)
into ADDF v0.1.0 on-disk trees that a downstream loader can consume.

Flow per record:
    unpack      → extract every archive recursively
    inventory   → walk tree, classify each file by format/modality/readability
    sniff       → rule-based layout matcher (ImageFolder, paired masks, ...)
    agent       → optional Gemma-4 E4B fallback for unusual layouts
    materialize → hardlink files into an ADDF directoryList tree
    validate    → sample-open one file per modality with the right reader

Default inputs:  ./data/downloads/{source}/{source_id}/  (envision-discovery layout)
Default outputs: ./data/actionable/{source}/{source_id}/

Part of the EyeACT project.
"""

from .pipeline import conform_record, conform_source

__version__ = "0.1.0"
__all__ = ["conform_record", "conform_source", "__version__"]
