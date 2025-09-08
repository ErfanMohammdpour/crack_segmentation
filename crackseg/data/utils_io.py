from __future__ import annotations

"""Utility I/O helpers for crack segmentation project.

This module provides:
- Path discovery for COCO annotations per split.
- JSON loading with helpful error messages.
- Seed utilities and device helpers.
"""

from pathlib import Path
import json
import logging
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def find_coco_annotation_file(split_dir: Path) -> Path:
    """Find a COCO annotation file within a split directory.

    Looks for files ending with "_annotations.coco.json" or "_annotations.coco.json" variations.
    Returns the first match by lexical order for determinism.
    """
    candidates = sorted(
        [p for p in split_dir.glob("*.json") if "annotations" in p.name.lower() and p.suffix == ".json"]
    )
    for p in candidates:
        if p.name.lower().endswith("_annotations.coco.json") or "coco" in p.name.lower():
            LOGGER.info("Using COCO annotation file: %s", p)
            return p
    if candidates:
        LOGGER.info("Falling back to first JSON file: %s", candidates[0])
        return candidates[0]
    raise FileNotFoundError(f"No COCO annotation JSON found in {split_dir}")


def load_json(path: Path) -> Dict:
    """Load JSON file with UTF-8 encoding and friendly errors."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON {path}: {e}") from e


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def get_device() -> torch.device:
    """CPU-only device helper (explicitly chooses CPU)."""
    return torch.device("cpu")


def resolve_paths_from_config(cfg: Dict) -> Tuple[Path, Path, Path]:
    """Resolve common root paths from config dict."""
    data_root = Path(cfg["DATA_ROOT"]).expanduser()
    runs_dir = Path(cfg["RUNS_DIR"]).expanduser()
    outputs_dir = Path(cfg["OUTPUTS_DIR"]).expanduser()
    return data_root, runs_dir, outputs_dir

