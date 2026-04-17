"""Agent fallback via local Gemma 4 E4B (llama.cpp).

When the rule-based sniffer returns a low-confidence Recipe, this module
asks a local LLM to propose one from the inventory + README + scraped
metadata. Runs on CPU, no API calls.

Install (optional — only needed when using the agent):
    pip install envision-eye-actionable[agent]

Then pass ``--agent-model <path-to-gguf>`` to ``envision-conform`` or pass
``AgentConfig(model_path=...)`` programmatically.

Recommended model: `gemma-4-E4B-it-Q4_K_M.gguf` from
https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF (~5 GB, 5-15 tok/s on CPU).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .inventory import Inventory, sample_text
from .sniff import Placement, Recipe

logger = logging.getLogger(__name__)

# Process-wide cache. A 5 GB GGUF + its context state takes ~7 GB of RAM to
# load. We don't want to reload it per record.
_LLAMA_CACHE: dict = {}


@dataclass
class AgentConfig:
    model_path: Path | None = None
    max_context_tokens: int = 8192      # input context size
    max_new_tokens: int = 1024          # output budget
    temperature: float = 0.2
    n_threads: int | None = None        # None → llama.cpp auto-detects CPU count
    verbose: bool = False


def propose_recipe(
    inv: Inventory,
    source_metadata: dict | None = None,
    config: AgentConfig | None = None,
) -> Recipe | None:
    """Ask the agent for a Recipe given inventory + README text + metadata.

    Returns None if the agent is disabled, the model fails to load, or the
    response can't be parsed into a valid Recipe.
    """
    config = config or AgentConfig()
    if config.model_path is None:
        return None
    model_path = Path(config.model_path).expanduser()
    if not model_path.exists():
        logger.warning(f"agent model not found: {model_path}")
        return None

    prompt = _build_prompt(inv, source_metadata)

    try:
        response = _call_llm(prompt, config, model_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"agent call failed: {e}")
        return None

    if not response:
        return None
    return _parse_response(response)


def _build_prompt(inv: Inventory, source_metadata: dict | None) -> str:
    readme = sample_text(inv.root, inv.readme_paths) if inv.readme_paths else ""
    meta_blob = ""
    if source_metadata:
        meta_blob = (
            f"Title: {source_metadata.get('title', '')}\n"
            f"Description: {(source_metadata.get('description') or '')[:1200]}\n"
            f"Keywords: {', '.join(source_metadata.get('keywords', [])[:15])}\n"
        )

    summary = inv.summary()
    # Compact the directory examples — just a few sample paths, not all files
    sample_paths = [str(f.path) for f in inv.files[:30]]
    sample_block = "\n".join(sample_paths)

    return f"""You are organizing an eye imaging dataset into the ADDF v0.1.0 directory format.

TASK: Propose a "recipe" — a mapping from source file globs to target ADDF directories.

Dataset metadata:
{meta_blob}
File inventory summary:
{json.dumps(summary, indent=2)}

Sample file paths (first {len(sample_paths)} of {len(inv.files)}):
{sample_block}

README excerpts:
{readme[:3000]}

VALID MODALITIES (pick the one that matches each file group):
  retinal_photography  — fundus color photos, .jpg/.png/.tif
  retinal_oct          — OCT B-scans and volumes (.fds/.fda/.e2e/.img/.oct/.vol/.dcm)
  retinal_imaging      — generic DICOM-format retinal imaging
  volumetric_imaging   — .nii, .nii.gz, .nrrd, .mha, .mhd
  tabular_data         — .csv/.tsv/.xlsx label files and clinical measurements
  derived_data         — .h5/.hdf5/.mat/.npy/.npz
  documentation        — .pdf/.docx/.txt/.md README/paper/label doc
  metadata             — .json/.xml/.yaml sidecar metadata

VALID ROLES:
  image, mask, label, documentation, raw

VALID DIRECTORY TYPES:
  modality, dataType, device

Return ONLY a JSON object with this exact shape (no prose before or after):

{{
  "task_type": "classification" | "segmentation" | "detection" | "raw",
  "placements": [
    {{"glob": "<source glob like 'images/**/*.jpg'>",
      "addf_dir": "<target e.g. 'retinal_photography/image'>",
      "modality": "<one of the modalities above>",
      "role": "<one of the roles above>",
      "directory_type": "<one of the directory types above>"}}
  ],
  "notes": ["<short observations about the dataset>"]
}}
"""


def _get_llama(model_path: Path, config: AgentConfig):
    """Load the GGUF once per process; cache by (path, context_size)."""
    key = (str(model_path), config.max_context_tokens)
    if key in _LLAMA_CACHE:
        return _LLAMA_CACHE[key]

    # Lazy import — llama-cpp-python is an optional dep
    from llama_cpp import Llama

    logger.info(f"loading agent model from {model_path} ({config.max_context_tokens} ctx)")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=config.max_context_tokens,
        n_threads=config.n_threads,
        verbose=config.verbose,
    )
    _LLAMA_CACHE[key] = llm
    return llm


def _call_llm(prompt: str, config: AgentConfig, model_path: Path) -> str | None:
    """Run a single-turn completion against the local GGUF."""
    llm = _get_llama(model_path, config)

    # Gemma chat template: <start_of_turn>user … <end_of_turn>\n<start_of_turn>model
    messages = [{"role": "user", "content": prompt}]
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _extract_json_block(text: str) -> str | None:
    """Pull a JSON object out of an LLM reply.

    Small models like to wrap output in ```json fences or add prose. We
    look for the first {...} span (balanced braces) and return that.
    """
    if not text:
        return None

    # Strip ```json fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Otherwise find the first top-level {…} via brace counting
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_response(response: str) -> Recipe | None:
    blob = _extract_json_block(response)
    if blob is None:
        logger.debug("agent: no JSON object found in response")
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        logger.debug(f"agent: JSON decode failed: {e}")
        return None

    placements_raw = data.get("placements", [])
    if not isinstance(placements_raw, list):
        return None

    placements = []
    for p in placements_raw:
        if not isinstance(p, dict) or "glob" not in p or "addf_dir" not in p:
            continue
        placements.append(Placement(
            glob=str(p["glob"]),
            addf_dir=str(p["addf_dir"]),
            modality=str(p.get("modality", "other")),
            role=str(p.get("role", "raw")),
            directory_type=str(p.get("directory_type", "modality")),
        ))

    if not placements:
        return None

    return Recipe(
        task_type=str(data.get("task_type", "raw")),
        placements=placements,
        confidence=0.7,  # Agent outputs get a fixed mid confidence
        notes=[str(n) for n in data.get("notes", []) if isinstance(n, (str, int, float))],
    )
