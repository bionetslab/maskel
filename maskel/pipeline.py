"""The three stages chained: mask -> voxel list -> cleaned voxel list -> features."""

from __future__ import annotations

from pathlib import Path

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
    voxels.attrs = {"stage": "skeletonize", "source": str(input_path)}
    _write(voxels, out_dir, name + SKELETON_SUFFIX, config)
    return voxels


def run_postprocess(
    voxels: VoxelList, out_dir: Path, config: PipelineConfig, name: str
) -> VoxelList:
    """Stage 2: write and return the cleaned-up voxel list."""
    cleaned = postprocess(voxels, config.postprocessing)
    cleaned.attrs = {**voxels.attrs, "stage": "postprocess"}
    _write(cleaned, out_dir, name + POSTPROCESSED_SUFFIX, config)
    return cleaned


def run_features(
    voxels: VoxelList, out_dir: Path, config: PipelineConfig, name: str
) -> None:
    """Stage 3: write the feature tables and graphs of *voxels*."""
    features = analyze(voxels, config.feature_extraction)
    write_features(out_dir, name, features, config.output)


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
    write_voxels(out_dir / f"{stem}.h5", voxels)
    if config.output.write_skeleton_mask:
        write_skeleton_mask(out_dir / f"{stem}_mask.h5", voxels)
    if config.output.write_skeleton_radii:
        write_skeleton_radii(out_dir / f"{stem}_radii.h5", voxels)
