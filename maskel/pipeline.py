"""The three stages chained: mask -> voxel list -> cleaned voxel list -> features."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from maskel.config import PipelineConfig
from maskel.features import analyze
from maskel.io.images import HDF5_EXTENSIONS, open_image
from maskel.io.tables import write_features
from maskel.io.voxels import (
    VoxelList,
    read_voxels,
    write_skeleton_mask,
    write_skeleton_radii,
    write_voxels,
)
from maskel.postprocessing import postprocess
from maskel.progress import format_duration
from maskel.skeletonize import skeletonize

SKELETON_SUFFIX = "_voxels"
POSTPROCESSED_SUFFIX = "_voxels_post"


def run(
    input_path: Path,
    out_dir: Path,
    config: PipelineConfig,
    *,
    dataset: str | None = None,
    tile_shape: int | tuple[int, ...] | str | None = None,
    halo: int | tuple[int, ...] | None = None,
    max_diameter: float | None = None,
    progress: str | bool = "off",
    jobs: int = 1,
) -> None:
    """Run all three stages on *input_path*, writing both voxel lists and the features."""
    name = Path(input_path).stem
    voxels = run_skeletonize(
        input_path,
        out_dir,
        config,
        name,
        dataset=dataset,
        tile_shape=tile_shape,
        halo=halo,
        max_diameter=max_diameter,
        progress=progress,
        jobs=jobs,
    )
    voxels = run_postprocess(voxels, out_dir, config, name)
    run_features(voxels, out_dir, config, name)


def run_skeletonize(
    input_path: Path,
    out_dir: Path,
    config: PipelineConfig,
    name: str,
    *,
    dataset: str | None = None,
    tile_shape: int | tuple[int, ...] | str | None = None,
    halo: int | tuple[int, ...] | None = None,
    max_diameter: float | None = None,
    progress: str | bool = "off",
    jobs: int = 1,
) -> VoxelList:
    """Stage 1: write and return the voxel list of the image at *input_path*."""
    spacing = config.spacing
    is_hdf5 = Path(input_path).suffix.lower() in HDF5_EXTENSIONS
    if tile_shape is not None and not is_hdf5:
        # checked before loading: other formats can only be read whole
        raise ValueError(
            f"Tiling needs an HDF5 input, got '{Path(input_path).name}'; "
            "convert it to HDF5 or leave out the tile shape."
        )
    start = time.monotonic()
    with open_image(input_path, dataset) as image:
        if spacing is not None and len(spacing) != image.ndim:
            raise ValueError(
                f"Spacing {spacing} has {len(spacing)} axes but the image has {image.ndim}."
            )
        voxels = skeletonize(
            image,
            config.preprocessing,
            spacing=spacing,
            tile_shape=tile_shape,
            halo=halo,
            max_diameter=max_diameter,
            progress=progress,
            jobs=jobs,
            source=(Path(input_path), dataset) if is_hdf5 else None,
        )
    with_skeleton = len(np.unique(voxels.label))
    _report(
        f"Skeletonized in {_since(start)}: {len(voxels.coords):,} skeleton voxels "
        f"in {with_skeleton:,} of {len(voxels.labels):,} label(s)."
    )
    voxels.attrs = {"stage": "skeletonize", "source": str(input_path)}
    _write(voxels, out_dir, name + SKELETON_SUFFIX, config)
    return voxels


def run_postprocess(
    voxels: VoxelList, out_dir: Path, config: PipelineConfig, name: str
) -> VoxelList:
    """Stage 2: write and return the cleaned-up voxel list."""
    start = time.monotonic()
    cleaned = postprocess(voxels, config.postprocessing)
    before, after = len(voxels.coords), len(cleaned.coords)
    change = (after - before) / before if before else 0.0
    _report(
        f"Postprocessed in {_since(start)}: {before:,} -> {after:,} skeleton "
        f"voxels ({change:+.1%})."
    )
    cleaned.attrs = {**voxels.attrs, "stage": "postprocess"}
    _write(cleaned, out_dir, name + POSTPROCESSED_SUFFIX, config)
    return cleaned


def run_features(
    voxels: VoxelList, out_dir: Path, config: PipelineConfig, name: str
) -> None:
    """Stage 3: write the feature tables and graphs of *voxels*."""
    start = time.monotonic()
    counts = {"labels": 0, "branches": 0}

    def counted(results):
        for result in results:
            counts["labels"] += 1
            counts["branches"] += len(result.branch_data)
            yield result

    features = counted(analyze(voxels, config.feature_extraction))
    written = write_features(out_dir, name, features, config.output)
    _report(
        f"Extracted features in {_since(start)}: {counts['labels']:,} of "
        f"{len(voxels.labels):,} label(s) with a graph, {counts['branches']:,} branches."
    )
    graphs = [p for p in written if p.suffix in (".graphml", ".pkl")]
    for path in written:
        if path not in graphs:
            _report(f"  Wrote '{path.name}' ({_file_size(path)}).")
    if graphs:
        _report(f"  Wrote {len(graphs):,} graph file(s).")


def load_voxels(path: Path) -> tuple[VoxelList, str]:
    """Read a voxel list and the image name it was written under."""
    stem = Path(path).stem
    for suffix in (POSTPROCESSED_SUFFIX, SKELETON_SUFFIX):
        stem = stem.removesuffix(suffix)
    return read_voxels(path), stem


def _write(voxels: VoxelList, out_dir: Path, stem: str, config: PipelineConfig) -> None:
    """Write *voxels* with the config that produced them, plus the configured skeleton images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    voxels.attrs["config"] = config.to_dict()
    dense = "x".join(str(n) for n in voxels.shape)
    outputs = [("voxel list", write_voxels, f"{stem}.h5")]
    if config.output.write_skeleton_mask:
        outputs.append((f"dense {dense} skeleton mask", write_skeleton_mask, f"{stem}_mask.h5"))
    if config.output.write_skeleton_radii:
        outputs.append((f"dense {dense} skeleton radii", write_skeleton_radii, f"{stem}_radii.h5"))
    for what, write, filename in outputs:
        start = time.monotonic()
        write(out_dir / filename, voxels)
        _report(
            f"  Wrote {what} '{filename}' "
            f"({_file_size(out_dir / filename)}, {_since(start)})."
        )


def _report(message: str) -> None:
    """Print *message* to the console right away."""
    print(message, flush=True)


def _since(start: float) -> str:
    """Time since *start* (a `time.monotonic` value), formatted."""
    return format_duration(time.monotonic() - start)


def _file_size(path: Path) -> str:
    """Size of the file at *path* in kB, MB or GB."""
    size = path.stat().st_size
    if size >= 1e9:
        return f"{size / 1e9:.2f} GB"
    if size >= 1e6:
        return f"{size / 1e6:.1f} MB"
    return f"{size / 1e3:.1f} kB"
