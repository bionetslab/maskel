"""`maskel skeletonize_tiled`: thin one large HDF5 image tile by tile."""

from __future__ import annotations

import argparse
from pathlib import Path

from maskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "skeletonize_tiled",
        help=(
            "Skeletonize one large image tile by tile. Writes only the "
            "skeleton - no features, no CSVs."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Single HDF5 file (.h5/.hdf5). Not a batch command: tiling "
        "targets one image that does not fit in memory, and HDF5 can be read "
        "block by block, compressed or not.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path of the image dataset inside the HDF5 file, e.g. "
        "'/volume/mask'. Optional when the file holds exactly one 2D or 3D "
        "dataset.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Work directory. Receives manifest.json, done.jsonl and "
        "<name>_skeleton.npy. Nothing is staged per tile.",
    )
    parser.add_argument(
        "--tile-shape",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Core tile edge length: one value for all axes, or one per axis. "
        "This sets peak memory: budget ~5 bytes per voxel of (N + 2*halo) per "
        "axis. Keep it well above --halo.",
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=None,
        help="Voxels read beyond each tile core for the thinning "
        "(default: 100). Must exceed the thinning's reach, roughly 8x the "
        "largest vessel radius in 3D. Opening, closing and hole filling add "
        "their own margin on top of this.",
    )
    parser.add_argument(
        "--config",
        help="Optional config JSON. Opening, closing and hole filling run per "
        "tile (fill_holes needs max_hole_size); anything after the thinning, "
        "such as junction cleanup, is rejected rather than ignored.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "lines", "off"),
        default="auto",
        help="Progress display. 'bar' rewrites one line in place, 'lines' "
        "prints a new line every few percent (for logs). 'auto' picks the "
        "bar on a terminal or under PyCharm, lines otherwise.",
    )
    parser.set_defaults(handler=run_skeletonize_tiled)


def run_skeletonize_tiled(args: argparse.Namespace) -> int:
    import warnings

    from maskel.tiling import (
        DEFAULT_HALO,
        BlockReader,
        TileGrid,
        analyze_tiled,
        preprocess_from_config,
    )
    from maskel.tiling.reader import HDF5_EXTENSIONS

    in_path = Path(args.input)
    if not in_path.is_file():
        raise ValueError(f"Input '{in_path}' is not an existing file.")
    if in_path.suffix.lower() not in HDF5_EXTENSIONS:
        raise ValueError(
            f"Unsupported input '{in_path.name}'. skeletonize_tiled reads "
            f"HDF5 only ({', '.join(sorted(HDF5_EXTENSIONS))}); convert the "
            f"image to HDF5 first."
        )
    config = (
        load_pipeline_config(Path(args.config))
        if args.config
        else PipelineConfig(extraction=ExtractionConfig(), output=OutputConfig())
    )

    halo = DEFAULT_HALO if args.halo is None else args.halo
    tile_shape = (
        args.tile_shape[0] if len(args.tile_shape) == 1 else tuple(args.tile_shape)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the header is read here - the volume itself is streamed tile by
    # tile, so nothing proportional to its size is ever resident.
    # Opening it here also validates the dataset and its compression filter
    # before anything is written.
    with BlockReader(in_path, args.dataset) as reader:
        shape = reader.shape

    # Report the geometry up front: the redundant-work factor is the number
    # people get wrong, and a bad one costs hours on a large volume. The grid
    # is rebuilt inside analyze_tiled; silence its duplicate warning here.
    preprocess = preprocess_from_config(config.extraction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = TileGrid(
            shape=shape, tile_shape=tile_shape, halo=halo + preprocess.margin
        )
    split = f" ({halo} thinning + {preprocess.margin} preprocessing)"
    print(
        f"{in_path.name}: shape={shape}, {len(grid)} tile(s) of "
        f"{grid.tile_shape} with halo={grid.halo}"
        f"{split if preprocess.margin else ''} "
        f"(reads {grid.read_overhead:.2f}x the volume)",
        flush=True,
    )

    analyze_tiled(
        in_path,
        in_path.stem,
        config,
        work_dir=out_dir,
        tile_shape=tile_shape,
        halo=halo,
        progress=args.progress,
        dataset=args.dataset,
    )

    print(f"Wrote '{out_dir / f'{in_path.stem}_skeleton.npy'}'.")
    return 0
