"""Image loading and result saving helpers."""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import itk
import numpy as np
from PIL import Image

from maskel.config import OutputConfig
from maskel.graphml import write_graphml
from maskel.pipeline import AnalysisResult


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".mhd":
        arr = np.asarray(itk.imread(str(path)))
    elif suffix in (".tif", ".tiff"):
        # Use tifffile rather than PIL: PIL reads only the *first frame* of a
        # multi-page/volumetric TIFF, silently turning a 3D stack into one
        # slice. tifffile loads the full volume.
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        with Image.open(path) as im:
            arr = np.asarray(im)

    if arr.ndim == 0:
        raise ValueError("Scalar input is not supported")

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.max(arr[..., :3], axis=-1)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape={arr.shape}")

    return arr


def sanitize_for_csv(value: object) -> object:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: sanitize_for_csv(v) for k, v in row.items()})


def save_skeleton(
    path: Path,
    skeleton: np.ndarray,
    *,
    npy: bool = True,
    png: bool = False,
) -> None:
    if npy:
        np.save(path.with_suffix(".npy"), skeleton.astype(np.uint8))
    if png:
        if skeleton.ndim != 2:
            print(
                "Warning: skipping PNG skeleton output for "
                f"{path.with_suffix('.png')}: only supported for 2D images",
                file=sys.stderr,
            )
        else:
            img = Image.fromarray((skeleton > 0).astype(np.uint8) * 255)
            img.save(path.with_suffix(".png"))


def save_radius(path: Path, radius_matrix: np.ndarray) -> None:
    np.save(path.with_suffix(".npy"), radius_matrix.astype(np.float64))


def save_analysis_outputs(
    output_dir: Path,
    base_name: str,
    result: AnalysisResult,
    config: OutputConfig,
    *,
    write_summary: bool = True,
) -> None:
    """Write all analysis results to disk according to *config*.

    Parameters
    ----------
    output_dir
        Parent directory under which a *base_name* subdirectory is created.
    base_name
        Used as subdirectory name and as prefix for all output files.
    result
        Analysis result to save.
    config
        Output configuration controlling which files are written.
    write_summary
        When True (default), also write a per-image summary CSV inside the
        output subdirectory.  Set to False in batch mode where the caller
        produces an aggregated summary.csv at the top level.
    """
    out = output_dir / base_name
    out.mkdir(parents=True, exist_ok=True)

    if config.write_skeleton_npy or config.write_skeleton_png:
        save_skeleton(
            out / f"{base_name}_skeleton",
            result.skeleton,
            npy=config.write_skeleton_npy,
            png=config.write_skeleton_png,
        )

    if config.write_radius and result.radius_matrix is not None:
        save_radius(out / f"{base_name}_radius", result.radius_matrix)

    if config.write_graphml:
        for obj in result.objects:
            if obj.graph is None or obj.branch_data is None:
                continue
            write_graphml(
                obj.graph,
                obj.branch_data,
                out / f"{base_name}_{obj.object_id}_graph.graphml",
                summary_features=obj.summary_features or None,
                radius_matrix=obj.radius_matrix,
            )

    if config.write_branch_csv and result.branch_records:
        write_csv(out / f"{base_name}_branches.csv", result.branch_records)

    if config.write_node_csv and result.node_records:
        write_csv(out / f"{base_name}_nodes.csv", result.node_records)

    if write_summary and config.write_summary_csv and result.summary_features:
        write_csv(
            out / f"{base_name}_summary.csv",
            [{"image": base_name, **row} for row in result.summary_features],
        )
