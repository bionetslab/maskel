"""`maskel run`: analyse a batch of images, optionally in parallel."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from maskel.config import PipelineConfig, load_pipeline_config

SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".mhd"}
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run batch analysis on one or more images using a config JSON.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config JSON (can be exported from napari).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for CSVs and optional skeletons.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for supported image files.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (0 = all CPU cores).",
    )
    parser.add_argument(
        "--spacing",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Per-axis physical pixel/voxel size (e.g. '--spacing 2.0 2.0' "
            "for 2D or '--spacing 1.0 0.5 0.5' for 3D), overriding the "
            "config's extraction.spacing for this whole batch run. Images "
            "whose dimensionality doesn't match fall back to isotropic "
            "spacing with a warning, rather than failing the batch."
        ),
    )
    parser.set_defaults(handler=run_batch)


def discover_input_paths(inputs: list[str], recursive: bool) -> list[Path]:
    paths: set[Path] = set()

    for raw in inputs:
        token = Path(raw)
        if token.exists():
            if token.is_file():
                if token.suffix.lower() in SUPPORTED_EXTENSIONS:
                    paths.add(token.resolve())
            elif token.is_dir():
                pattern = "**/*" if recursive else "*"
                for p in token.glob(pattern):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        paths.add(p.resolve())
            continue

        for match in glob.glob(raw, recursive=recursive):
            p = Path(match)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.add(p.resolve())

    return sorted(paths)


def compute_safe_names(input_paths: list[Path]) -> list[str]:
    name_counts: dict[str, int] = {}
    safe_names: list[str] = []
    for p in input_paths:
        base = p.stem
        seen = name_counts.get(base, 0)
        name_counts[base] = seen + 1
        safe_names.append(base if seen == 0 else f"{base}_{seen + 1}")
    return safe_names


def process_one(
    in_path: Path,
    safe_name: str,
    out_dir: Path,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    """Load, analyse, save one image. Returns one summary row per object for agg CSV.

    Also the worker function for `--jobs`, so it must stay picklable by
    reference (module-level).
    """
    from maskel._io import load_image, save_analysis_outputs
    from maskel.pipeline import analyze_segmentation_mask

    image = load_image(in_path)
    result = analyze_segmentation_mask(mask=image, config=config)

    save_analysis_outputs(
        out_dir, safe_name, result, config.output, write_summary=False
    )

    return [{"image": in_path.name, **row} for row in result.summary_features]


def run_batch(args: argparse.Namespace) -> int:
    from maskel._io import write_csv

    config = load_pipeline_config(Path(args.config))
    if args.spacing is not None:
        config.extraction.spacing = tuple(args.spacing)
    input_paths = discover_input_paths(args.input, recursive=args.recursive)
    if not input_paths:
        raise ValueError("No input files found. Check --input and --recursive.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_names = compute_safe_names(input_paths)
    jobs = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    if jobs < 1:
        raise ValueError("--jobs must be 0 (auto) or a positive integer")
    total = len(input_paths)

    if jobs == 1:
        summary_rows = []
        for idx, (in_path, safe_name) in enumerate(zip(input_paths, safe_names), 1):
            print(f"[{idx}/{total}] {in_path.name}", flush=True)
            summary_rows.extend(process_one(in_path, safe_name, out_dir, config))
    else:
        summary_rows = run_parallel(
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


def run_parallel(
    input_paths: list[Path],
    safe_names: list[str],
    out_dir: Path,
    config: PipelineConfig,
    jobs: int,
    total: int,
) -> list[dict[str, object]]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
                summary_rows.extend(fut.result())
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
        # Snapshot the workers first; shutdown() drops the executor's handles.
        workers = mp.active_children()
        ex.shutdown(wait=False, cancel_futures=True)
        for w in workers:
            w.kill()
        for w in workers:
            w.join()
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
