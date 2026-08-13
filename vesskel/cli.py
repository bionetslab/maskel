"""Standalone CLI for VesSkel batch analysis."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from vesskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesskel",
        description="VesSkel CLI for batch-vessel-analysis.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"vesskel {version('vesskel')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run batch analysis on one or more images using a config JSON.",
    )
    run_parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files, directories, or glob patterns.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config JSON (can be exported from napari).",
    )
    run_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for CSVs and optional skeletons.",
    )
    run_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for supported image files.",
    )
    run_parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (0 = all CPU cores).",
    )

    skel_parser = subparsers.add_parser(
        "skeletonize_tiled",
        help=(
            "Skeletonize one large image tile by tile. Writes only the "
            "skeleton - no features, no CSVs."
        ),
    )
    skel_parser.add_argument(
        "--input",
        required=True,
        help="Single input image. Not a batch command: tiling targets one "
        "image that does not fit in memory.",
    )
    skel_parser.add_argument(
        "--out",
        required=True,
        help="Work directory. Receives manifest.json, done.jsonl and "
        "<name>_skeleton.npy. Nothing is staged per tile.",
    )
    skel_parser.add_argument(
        "--tile-shape",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Core tile edge length: one value for all axes, or one per axis. "
        "This sets peak memory: budget ~5 bytes per voxel of (N + 2*halo) per "
        "axis, times --jobs. Keep it well above --halo.",
    )
    skel_parser.add_argument(
        "--halo",
        type=int,
        default=None,
        help="Voxels read beyond each tile core (default: 100). Must exceed "
        "the thinning's reach, roughly 8x the largest vessel radius in 3D.",
    )
    skel_parser.add_argument(
        "--config",
        help="Optional config JSON. This command only thins, so a config "
        "requesting preprocessing or junction cleanup is rejected rather "
        "than ignored.",
    )
    skel_parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (0 = all CPU cores).",
    )
    skel_parser.add_argument(
        "--progress",
        choices=("auto", "bar", "lines", "off"),
        default="auto",
        help="Progress display. 'bar' rewrites one line in place, 'lines' "
        "prints a new line every few percent (for logs). 'auto' picks the "
        "bar on a terminal or under PyCharm, lines otherwise.",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Create a starter config JSON.",
    )
    init_parser.add_argument(
        "out", nargs="?", default="vesskel.json", help="Output config path."
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate and print a normalised config JSON.",
    )
    validate_parser.add_argument(
        "config", nargs="?", default="vesskel.json", help="Config JSON path."
    )

    completions_parser = subparsers.add_parser(
        "completions",
        help="Print shell completion script to stdout.",
    )
    completions_parser.add_argument(
        "shell",
        choices=("bash", "zsh", "powershell"),
        help="Target shell.",
    )

    return parser


# WARNING: keep this guard and everything above it free of heavy imports (numpy, PIL, vesskel.pipeline, vesskel._batch).
# The guard lets shell completions exit in 81 ms instead of 800 ms.
# Tests in tests/test_cli.py::TestCompletionSpeed enforce this.
if "_ARGCOMPLETE" in os.environ or (len(sys.argv) > 1 and sys.argv[1] == "completions"):
    if "_ARGCOMPLETE" in os.environ:
        try:
            from argcomplete import autocomplete

            autocomplete(_make_parser())
        except ImportError:
            pass
    else:
        try:
            from argcomplete import shellcode

            shell = sys.argv[2] if len(sys.argv) > 2 else "zsh"
            print(shellcode(["vesskel"], shell=shell))
        except ImportError:
            pass
    sys.exit(0)


_SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".mhd"}
)


def _parse_args() -> argparse.Namespace:
    parser = _make_parser()
    try:
        from argcomplete import autocomplete

        autocomplete(parser)
    except ImportError:
        pass
    return parser.parse_args()


def _discover_input_paths(inputs: list[str], recursive: bool) -> list[Path]:
    paths: set[Path] = set()

    for raw in inputs:
        token = Path(raw)
        if token.exists():
            if token.is_file():
                if token.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    paths.add(token.resolve())
            elif token.is_dir():
                pattern = "**/*" if recursive else "*"
                for p in token.glob(pattern):
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                        paths.add(p.resolve())
            continue

        for match in glob.glob(raw, recursive=recursive):
            p = Path(match)
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                paths.add(p.resolve())

    return sorted(paths)


def _compute_safe_names(input_paths: list[Path]) -> list[str]:
    name_counts: dict[str, int] = {}
    safe_names: list[str] = []
    for p in input_paths:
        base = p.stem
        seen = name_counts.get(base, 0)
        name_counts[base] = seen + 1
        safe_names.append(base if seen == 0 else f"{base}_{seen + 1}")
    return safe_names


def _run_batch(args: argparse.Namespace) -> int:
    from vesskel._batch import process_one
    from vesskel._io import write_csv

    config = load_pipeline_config(Path(args.config))
    input_paths = _discover_input_paths(args.input, recursive=args.recursive)
    if not input_paths:
        raise ValueError("No input files found. Check --input and --recursive.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_names = _compute_safe_names(input_paths)
    jobs = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    if jobs < 1:
        raise ValueError("--jobs must be 0 (auto) or a positive integer")
    total = len(input_paths)

    if jobs == 1:
        summary_rows = []
        for idx, (in_path, safe_name) in enumerate(zip(input_paths, safe_names), 1):
            print(f"[{idx}/{total}] {in_path.name}", flush=True)
            summary_rows.append(process_one(in_path, safe_name, out_dir, config))
    else:
        summary_rows = _run_parallel(
            input_paths, safe_names, out_dir, config, jobs, total
        )

    summary_rows.sort(key=lambda r: str(r.get("image", "")))

    if config.output.write_summary_csv:
        write_csv(out_dir / "summary.csv", summary_rows)

    print(
        f"Processed {len(input_paths)} image(s) with config '{args.config}'. "
        f"Outputs written to '{out_dir}'."
    )
    return 0


def _run_parallel(
    input_paths: list[Path],
    safe_names: list[str],
    out_dir: Path,
    config: PipelineConfig,
    jobs: int,
    total: int,
) -> list[dict[str, object]]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from vesskel._batch import process_one

    summary_rows: list[dict[str, object]] = []
    errors: list[tuple[int, str, Exception]] = []
    ctx = mp.get_context("spawn")

    proc = "process" if jobs == 1 else "processes"
    print(f"Spawning {jobs} worker {proc}...", flush=True)

    ex = ProcessPoolExecutor(max_workers=jobs, mp_context=ctx)
    futures = {
        ex.submit(process_one, p, sn, out_dir, config): (idx, p.name)
        for idx, (p, sn) in enumerate(zip(input_paths, safe_names), 1)
    }

    interrupted = False
    try:
        for fut in as_completed(futures):
            idx, name = futures[fut]
            try:
                summary_rows.append(fut.result())
                print(f"[{idx}/{total}] {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{idx}/{total}] {name} FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                errors.append((idx, name, exc))
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Killing workers...", flush=True)
        ex.kill_workers()
        print("Shutdown complete.", flush=True)
    finally:
        if not interrupted:
            ex.shutdown(wait=True)

    if errors:
        failures = "\n".join(
            f"  [{idx}/{total}] {name}: {exc}" for idx, name, exc in errors
        )
        raise RuntimeError(f"{len(errors)}/{total} image(s) failed:\n{failures}")

    return summary_rows


def _run_skeletonize_tiled(args: argparse.Namespace) -> int:
    import warnings

    from vesskel.tiling import DEFAULT_HALO, BlockReader, TileGrid, analyze_tiled

    in_path = Path(args.input)
    if not in_path.is_file():
        raise ValueError(f"Input '{in_path}' is not an existing file.")
    if in_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input '{in_path.name}'. Expected one of: "
            f"{', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )
    if args.jobs < 0:
        raise ValueError("--jobs must be 0 (auto) or a positive integer")

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
    shape = BlockReader(in_path).shape

    # Report the geometry up front: the redundant-work factor is the number
    # people get wrong, and a bad one costs hours on a large volume. The grid
    # is rebuilt inside analyze_tiled; silence its duplicate warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = TileGrid(shape=shape, tile_shape=tile_shape, halo=halo)
    print(
        f"{in_path.name}: shape={shape}, {len(grid)} tile(s) of "
        f"{grid.tile_shape} with halo={halo} "
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
        jobs=args.jobs,
        progress=args.progress,
    )

    print(f"Wrote '{out_dir / f'{in_path.stem}_skeleton.npy'}'.")
    return 0


def _config_init(args: argparse.Namespace) -> int:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        extraction=ExtractionConfig(),
        output=OutputConfig(),
    )
    with path.open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Wrote starter config to '{path}'.")
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_pipeline_config(Path(args.config))
    print(json.dumps(config.to_dict(), indent=2))
    print("Configuration is valid.")
    return 0


def _completions(args: argparse.Namespace) -> int:
    from argcomplete import shellcode

    print(shellcode(["vesskel"], shell=args.shell))
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return _run_batch(args)
    if args.command == "skeletonize_tiled":
        return _run_skeletonize_tiled(args)
    if args.command == "init":
        return _config_init(args)
    if args.command == "validate":
        return _validate_config(args)
    if args.command == "completions":
        return _completions(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
