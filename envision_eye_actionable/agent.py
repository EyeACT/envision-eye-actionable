"""Agent fallback (stub).

When the rule-based sniffer can't confidently match a layout, this module
will call a local Gemma 4 E4B model via llama.cpp to propose a recipe.

Not yet implemented — the sniffer is used alone in the MVP. This module
defines the interface so the caller can wire the agent in later without
reshuffling.

To enable:
  1. Install llama-cpp-python (CPU build): pip install llama-cpp-python
  2. Download a Gemma 4 E4B GGUF: https://huggingface.co/google/gemma-4-e4b-it-gguf
  3. Implement _call_llm() below to invoke the model with the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .inventory import Inventory, sample_text
from .sniff import Recipe


@dataclass
class AgentConfig:
    model_path: Path | None = None
    max_context_chars: int = 6000
    max_new_tokens: int = 1024
    temperature: float = 0.2


def propose_recipe(
    inv: Inventory,
    source_metadata: dict | None = None,
    config: AgentConfig | None = None,
) -> Recipe | None:
    """Ask the agent for a Recipe given inventory + README text + metadata.

    Returns None if the agent is not available or cannot propose a recipe.
    """
    config = config or AgentConfig()
    if config.model_path is None or not config.model_path.exists():
        return None  # Agent disabled

    prompt = _build_prompt(inv, source_metadata)
    response = _call_llm(prompt, config)
    if not response:
        return None

    return _parse_response(response)


def _build_prompt(inv: Inventory, source_metadata: dict | None) -> str:
    readme = sample_text(inv.root, inv.readme_paths) if inv.readme_paths else ""
    meta_blob = ""
    if source_metadata:
        meta_blob = (
            f"Title: {source_metadata.get('title', '')}\n"
            f"Description: {(source_metadata.get('description') or '')[:1000]}\n"
            f"Keywords: {', '.join(source_metadata.get('keywords', [])[:15])}\n"
        )

    summary = inv.summary()
    return f"""You are converting an eye imaging dataset into the ADDF v0.1.0 directory format.

Dataset metadata:
{meta_blob}

File inventory summary:
{summary}

README excerpts:
{readme[:3000]}

Propose a conversion recipe as JSON with this exact shape:
{{
  "task_type": "classification" | "segmentation" | "detection" | "raw",
  "placements": [
    {{"glob": "<source glob>", "addf_dir": "<target dir>", "modality": "<retinal_photography|retinal_oct|...>", "role": "image|mask|label|documentation|raw", "directory_type": "modality|dataType|device"}}
  ],
  "notes": ["..."]
}}

Respond with only the JSON object, no prose.
"""


def _call_llm(prompt: str, config: AgentConfig) -> str | None:
    """Not yet implemented. Wire llama-cpp-python here."""
    # Intentionally unimplemented. See module docstring.
    return None


def _parse_response(response: str) -> Recipe | None:
    import json
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None

    from .sniff import Placement, Recipe

    placements = [
        Placement(
            glob=p["glob"],
            addf_dir=p["addf_dir"],
            modality=p["modality"],
            role=p.get("role", "raw"),
            directory_type=p.get("directory_type", "modality"),
        )
        for p in data.get("placements", [])
    ]
    return Recipe(
        task_type=data.get("task_type", "raw"),
        placements=placements,
        confidence=0.7,  # Agent outputs get a fixed mid confidence
        notes=data.get("notes", []),
    )
