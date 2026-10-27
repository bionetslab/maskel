"""Config-aware entry point for tiled skeletonization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maskel.config import ExtractionConfig, PipelineConfig
from maskel.tiling.grid import DEFAULT_HALO
from maskel.tiling.preprocess import BlockPreprocess
from maskel.tiling.stream import skeletonize_streaming


def preprocess_from_config(extraction: ExtractionConfig) -> BlockPreprocess:
    """The preprocessing steps the tiled path runs on every block."""
    return BlockPreprocess(
        opening_radius=extraction.opening_radius,
        closing_iterations=extraction.closing_iterations,
        fill_holes=extraction.fill_holes,
        max_hole_size=extraction.max_hole_size,
    )


def _unsupported_steps(extraction: ExtractionConfig) -> list[str]:
    """Configured pipeline steps the tiled path does not run."""
    configured = {
        "junction_cleanup": extraction.junction_cleanup,
        "prune_spurs": extraction.prune_spurs,
        "spacing": extraction.spacing is not None,
    }
    return [name for name, is_set in configured.items() if is_set]


def analyze_tiled(
    source: Path | str | np.ndarray,
    base_name: str,
    config: PipelineConfig,
    *,
    work_dir: Path,
    tile_shape: int | tuple[int, ...],
    halo: int = DEFAULT_HALO,
    progress: str | bool = "off",
    resume: bool = True,
) -> None:
    """Thin *source* into ``work_dir/<base_name>_skeleton.npy``.

    Opening, closing and bounded hole filling run per block; nothing after the thinning
    does - no junction cleanup, spur pruning, radii, graph or features.
    Configuring any of those raises instead of being silently skipped. Every
    nonzero value is foreground, so an instance segmentation is thinned as one
    mask.
    """
    unsupported = _unsupported_steps(config.extraction)
    if unsupported:
        raise NotImplementedError(
            f"The tiled path stops after thinning, but the config sets: "
            f"{', '.join(unsupported)}. Disable them, or use "
            f"maskel.pipeline.analyze_segmentation_mask on a smaller input."
        )

    skeletonize_streaming(
        source,
        work_dir,
        tile_shape=tile_shape,
        halo=halo,
        preprocess=preprocess_from_config(config.extraction),
        progress=progress,
        out_path=Path(work_dir) / f"{base_name}_skeleton.npy",
        resume=resume,
    )
