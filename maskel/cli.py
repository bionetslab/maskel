"""Standalone CLI for Maskel batch analysis."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from maskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskel",
        description="Maskel CLI for batch-vessel-analysis.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"maskel {version('maskel')}",
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

    init_parser = subparsers.add_parser(
        "init",
        help="Create a starter config JSON.",
    )
    init_parser.add_argument(
        "out", nargs="?", default="maskel.json", help="Output config path."
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate and print a normalised config JSON.",
    )
    validate_parser.add_argument(
        "config", nargs="?", default="maskel.json", help="Config JSON path."
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


# WARNING: keep this guard and everything above it free of heavy imports (numpy, PIL, maskel.pipeline, maskel._batch).
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
            print(shellcode(["maskel"], shell=shell))
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
    from maskel._batch import process_one
    from maskel._io import write_csv

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
            summary_rows.extend(process_one(in_path, safe_name, out_dir, config))
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

    from maskel._batch import process_one

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

    print(shellcode(["maskel"], shell=args.shell))
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return _run_batch(args)
    if args.command == "init":
        return _config_init(args)
    if args.command == "validate":
        return _validate_config(args)
    if args.command == "completions":
        return _completions(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
