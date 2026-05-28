"""Shared data structures for conformer recipes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Placement:
    glob: str                # source glob relative to the unpacked root
    addf_dir: str            # target directory inside the ADDF tree
    modality: str            # retinal_photography, retinal_oct, etc.
    role: str                # "image", "mask", "label", "documentation", "raw"
    directory_type: str = "modality"  # modality / dataType / device


@dataclass
class Recipe:
    task_type: str                 # classification / segmentation / detection / raw
    placements: list[Placement] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
