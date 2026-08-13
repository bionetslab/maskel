"""Config-aware entry point for tiled skeletonization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from vesskel.config import ExtractionConfig, PipelineConfig
from vesskel.tiling.grid import DEFAULT_HALO
from vesskel.tiling.stream import skeletonize_streaming

# Everything `analyze_binary_image` does around thinning. The tiled path runs
# thinning and nothing else, so any of these being configured would silently
# yield a different skeleton than the monolithic path - which is worth an
# error rather than a footnote. Each entry is (config field, is-it-set,
# what it would do).
_UNSUPPORTED_STEPS: tuple[tuple[str, Callable[[ExtractionConfig], bool], str], ...] = (
    (
        "closing_iterations",
        lambda e: e.closing_iterations > 0,
        "morphological closing before thinning",
    ),
    (
        "fill_holes",
        lambda e: e.fill_holes,
        "hole filling before thinning - whether a background region is "
        "enclosed depends on the whole volume, so no single tile can decide it",
    ),
    (
        "max_hole_size",
        lambda e: e.max_hole_size > 0,
        "size-limited hole filling before thinning",
    ),
    (
        "junction_cleanup",
        lambda e: e.junction_cleanup,
        "collapsing triangle junctions after thinning",
    ),
)


def analyze_tiled(
    source: Path | str | np.ndarray,
    base_name: str,
    config: PipelineConfig,
    *,
    work_dir: Path,
    tile_shape: int | tuple[int, ...],
    halo: int = DEFAULT_HALO,
    jobs: int = 1,
    progress: str | bool = "off",
    resume: bool = True,
) -> None:
    """Skeletonize an image too large to thin - or even to load - in one piece.

    Streams *source* tile by tile into ``work_dir/<base_name>_skeleton.npy``
    (see `vesskel.tiling.skeletonize_streaming`). Nothing is returned and
    nothing is held in RAM but a single tile, so the limit is the tile size,
    not the volume.

    This runs the thinning and stops. There is no preprocessing, no junction
    cleanup, no radii, no graph and no features - those all need the whole
    volume at once, so they stay in `vesskel.pipeline.analyze_binary_image`.
    Configuring any of the mask- or skeleton-altering steps raises
    `NotImplementedError` rather than being ignored; the extraction and
    output settings are simply unused, since nothing past the skeleton is
    produced.

    The result is identical to `analyze_binary_image`'s skeleton for a
    default config, provided *halo* covers the thinning's domain of
    dependence (see `vesskel.tiling`).
    """
    configured = [
        f"  - {name}: {effect}"
        for name, is_set, effect in _UNSUPPORTED_STEPS
        if is_set(config.extraction)
    ]
    if configured:
        raise NotImplementedError(
            "The tiled path thins and nothing else, but the config asks for:\n"
            + "\n".join(configured)
            + "\nDisable them, or run vesskel.pipeline.analyze_binary_image on "
            "an input that fits in memory."
        )

    skeletonize_streaming(
        source,
        work_dir,
        tile_shape=tile_shape,
        halo=halo,
        jobs=jobs,
        progress=progress,
        out_path=Path(work_dir) / f"{base_name}_skeleton.npy",
        resume=resume,
    )
