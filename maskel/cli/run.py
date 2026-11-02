"""`maskel run`, `skeletonize`, `postprocess` and `features`: all stages or one at a time."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from maskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
)


def add_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the run command and one command per stage."""
    run = subparsers.add_parser(
        "run", help="Run all stages on one image: voxel lists and features."
    )
    _add_image_args(run)
    _add_common_args(run)
    run.set_defaults(handler=run_all)

    skeletonize = subparsers.add_parser(
        "skeletonize", help="Stage 1: image -> voxel list (<name>_voxels.h5)."
    )
    _add_image_args(skeletonize)
    _add_common_args(skeletonize)
    skeletonize.set_defaults(handler=run_skeletonize)

    postprocess = subparsers.add_parser(
        "postprocess",
        help="Stage 2: voxel list -> cleaned voxel list (<name>_voxels_post.h5).",
    )
    postprocess.add_argument("--input", required=True, help="Voxel list (.h5).")
    _add_common_args(postprocess)
    postprocess.set_defaults(handler=run_postprocess)

    features = subparsers.add_parser(
        "features", help="Stage 3: voxel list -> feature tables and graphs."
    )
    features.add_argument("--input", required=True, help="Voxel list (.h5).")
    _add_common_args(features)
    features.set_defaults(handler=run_features)


def _add_image_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for commands that read an image."""
    parser.add_argument(
        "--input", required=True, help="Image file (HDF5, TIFF, PNG, NPY, MHD, ...)."
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset path inside an HDF5 input."
    )
    parser.add_argument(
        "--spacing",
        nargs="+",
        type=float,
        default=None,
        help="Per-axis voxel size, overriding the config's extraction.spacing.",
    )
    parser.add_argument(
        "--tile-shape",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Tile core size, one value or one per axis. Omit for a single tile; "
        "only HDF5 input is read tile by tile.",
    )
    parser.add_argument(
        "--halo",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Voxels read beyond each tile core for the thinning, one value or "
        "one per axis (default: 100). Roughly 8x the largest radius in voxels.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Tiles processed in parallel (HDF5 input only; 0 = all cores). "
        "Memory grows with each job by about one tile.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "lines", "off"),
        default="auto",
        help="Tile progress display.",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments every command takes."""
    parser.add_argument("--config", default=None, help="Pipeline config JSON.")
    parser.add_argument("--out", required=True, help="Output directory.")


def _config(args: argparse.Namespace) -> PipelineConfig:
    """The config from --config (defaults without one), with --spacing applied."""
    config = (
        load_pipeline_config(Path(args.config))
        if args.config
        else PipelineConfig(extraction=ExtractionConfig(), output=OutputConfig())
    )
    if getattr(args, "spacing", None) is not None:
        config.extraction.spacing = tuple(args.spacing)
    return config


def _tiling(args: argparse.Namespace) -> dict:
    """Tiling keyword arguments from --tile-shape and --halo."""
    from maskel.tiling import DEFAULT_HALO

    def unpack(values):
        return values[0] if len(values) == 1 else tuple(values)

    return {
        "dataset": args.dataset,
        "tile_shape": None if args.tile_shape is None else unpack(args.tile_shape),
        "halo": DEFAULT_HALO if args.halo is None else unpack(args.halo),
        "progress": args.progress,
        "jobs": (os.cpu_count() or 1) if args.jobs == 0 else args.jobs,
    }


def run_all(args: argparse.Namespace) -> int:
    """Handler of `maskel run`."""
    from maskel.pipeline import run

    run(Path(args.input), Path(args.out), _config(args), **_tiling(args))
    print(f"Outputs written to '{args.out}'.")
    return 0


def run_skeletonize(args: argparse.Namespace) -> int:
    """Handler of `maskel skeletonize`."""
    from maskel import pipeline

    path = Path(args.input)
    pipeline.run_skeletonize(
        path, Path(args.out), _config(args), path.stem, **_tiling(args)
    )
    return 0


def run_postprocess(args: argparse.Namespace) -> int:
    """Handler of `maskel postprocess`."""
    from maskel import pipeline

    voxels, name = pipeline.load_voxels(Path(args.input))
    pipeline.run_postprocess(voxels, Path(args.out), _config(args), name)
    return 0


def run_features(args: argparse.Namespace) -> int:
    """Handler of `maskel features`."""
    from maskel import pipeline

    voxels, name = pipeline.load_voxels(Path(args.input))
    pipeline.run_features(voxels, Path(args.out), _config(args), name)
    return 0
