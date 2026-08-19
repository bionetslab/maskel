# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Maskel: 2D/3D vessel skeletonization and graph-based phenotype analysis (retinal fundus images, and other tubular structures like brain vasculature). This repo is the core algorithm package and CLI only.

Two companion repos consume this package:
- [napari-maskel](https://github.com/bionetslab/napari-maskel) — the napari plugin (widget + layer visualization), depends on `maskel` from PyPI.
- [maskel-evaluations](https://github.com/bionetslab/maskel-evaluations) — benchmarks, the HRF dataset, and analysis notebooks. Not a package; consumes `maskel` from PyPI.

## Commands

```sh
uv sync --extra dev              # install core + test tools

uv run pytest                    # all tests
uv run pytest -m "not slow"      # skip the 3D regression + skimage-comparison tests
uv run pytest tests/test_spur_pruning.py::test_name  # single test
uv run pytest --update-baseline  # regenerate regression baselines in tests/skeletons/ and tests/features/

uvx ruff check                   # lint (no repo-specific ruff config; defaults apply)
uvx ruff format --check          # format check
uv build                         # sdist/wheel build, run before release

maskel init config.json
maskel validate config.json
maskel run --input /path/to/images --config config.json --out outputs --jobs 0
```

## Architecture

**One pipeline, deliberately napari-free.** [maskel/pipeline.py](maskel/pipeline.py)'s `analyze_segmentation_mask()` is the single place the full per-image algorithm runs. It returns a plain-data `AnalysisResult` (skeleton, per-object results, radius matrix) with no napari dependency anywhere, not even under `TYPE_CHECKING`. The CLI's [maskel/_batch.py](maskel/_batch.py) (used across a `ProcessPoolExecutor`) calls it directly; napari-maskel calls the same function and builds its own `LayerDataTuple`s on top of the result in its own package. Never reintroduce a napari import here — that coupling was deliberately removed when the napari plugin was split into its own repo.

**Per-object splitting happens before the per-object pipeline runs.** `analyze_segmentation_mask()` first calls `_iter_object_crops()`, which treats a plain binary mask as one implicit object (id `1`) or, when the input has more than one distinct nonzero value, treats each value as its own object (an instance segmentation map). Each object is cropped to its own bounding box (padded by 1px, clamped to the array) and run through `_analyze_single_object()` — the actual per-object pipeline, described below — independently. This is why touching-but-distinct labeled objects end up correctly skeletonized as separate structures rather than merged into one: `mask == object_id` never includes a neighboring object's pixels, regardless of adjacency. `skeleton`/`radius_matrix`/`preprocessed_binary` are stitched back into full-image-sized arrays (via `np.maximum`, not plain assignment — two objects' 1px-padded crops can overlap at a shared border) since more than one consumer (napari's display, the CLI's `.npy` output) wants a whole-image view and that stitching work shouldn't be duplicated in each of them.

**`AnalysisResult.objects` is a list of `ObjectResult`, one per object — not several parallel lists.** Each `ObjectResult` bundles everything about one object: `object_id` (the mask's own label value), its crop `offset`, and its already-tagged, already-globally-positioned `summary_features`/`branch_records`/`node_records`, plus its own local-coordinate `graph`/`branch_data`/`radius_matrix` (needed by napari-maskel, to place branch/node layers in global coordinates without a single unified graph spanning touching-but-distinct objects, and by graphml export — one file per object, since skan/networkx have no notion of a multi-object disjoint graph). Earlier this was spread across four separately-indexed lists correlated only by a shared `object_id` key; that's exactly the kind of parallel-lists-that-must-stay-in-sync structure to avoid reintroducing. `AnalysisResult.summary_features`/`branch_records`/`node_records` still exist as read-only properties that flatten `objects` for convenience (e.g. writing one CSV across all objects) — don't add real fields with those names back.

Pipeline stages in `_analyze_single_object` (run once per object), in order, each individually toggled by `ExtractionConfig`:
1. optional morphological preprocessing (`preprocess_binary`: closing, hole filling)
2. `lee94_thin` (dispatches 2D/3D by `img.ndim`, see [maskel/thin.py](maskel/thin.py)) → skeleton
3. optional `collapse_triangle_junctions` (junction cleanup, needs an EDT radius matrix computed on the *original* skeleton)
4. `build_vessel_graph` → `skan.Skeleton`
5. optional spur pruning (`prune_short_spurs`), repeated up to `spur_iterations` times since removing one spur can expose another — the graph is rebuilt from scratch after each pass because pruning mutates skeleton pixels
6. optional radius estimation (EDT-based, `compute_radii` / `per_segment_radii`)
7. branch summary via `skan.summarize`, augmented with `tortuosity`/`straightness`
8. summary feature extraction (`extract_vessel_features`), branch/node record extraction

**Config is the contract with napari-maskel.** [maskel/config.py](maskel/config.py) defines `ExtractionConfig`/`OutputConfig`/`PipelineConfig` as the single schema (`CONFIG_SCHEMA_VERSION`, currently 3) this package and napari-maskel both read/write. A config exported from the napari plugin (**Save Config**) is the same JSON the CLI consumes (`maskel run --config`). Any change to extraction/output options must be threaded through `to_dict`/`from_dict` on all three dataclasses and the README config table, with the schema version bumped if the shape changes. Unknown keys warn rather than error (`_warn_unknown_keys`), so old configs stay loadable across additive changes.

**Thinning implementations are algorithmically separate from everything else.** [maskel/thin_2d.py](maskel/thin_2d.py) and [maskel/thin_3d.py](maskel/thin_3d.py) implement Lee et al. 1994 thinning (2D uses a lookup table of simple/endpoint conditions; 3D uses numba-jitted Euler-invariance + simple-point checks with parallel sub-iterations over 6 border directions). These are performance-critical; [docs/euler_redundancy.typ](docs/euler_redundancy.typ) proves that the Euler-invariant check is redundant once a 2D image is embedded as a 1-voxel-thick 3D volume, which is why `thin_2d.py` can skip it. Benchmarks against `VesselVio` and `skimage.morphology.skeletonize` live in `maskel-evaluations`; the self-contained skimage comparison (brain volume, no external dataset) stays here as `tests/test_3d_skimage_comparison.py`.

**CLI startup speed is a hard constraint.** The shell-completion fast path in [maskel/cli.py](maskel/cli.py) (top of file, guarded by `_ARGCOMPLETE` env var / `sys.argv[1] == "completions"`) must stay free of heavy imports (numpy, PIL, `maskel.pipeline`, `maskel._batch`) — `tests/test_cli.py::TestCompletionSpeed` enforces the latency budget. Only import those inside the functions that actually need them (`_run_batch`, `_run_parallel`), not at module scope.

**Batch parallelism**: `maskel run --jobs N` spawns a `ProcessPoolExecutor` (spawn context) over `_batch.process_one`, one call per image; `--jobs 0` uses all CPU cores. `Ctrl-C` triggers a clean worker kill rather than hanging on shutdown — preserve that behavior if touching `_run_parallel` in [maskel/cli.py](maskel/cli.py).

**Feature glossary**: domain terms used throughout `features.py`/branch data columns (tortuosity, straightness, hgu, fractal dimension, lacunarity, etc.) are defined in [docs/glossary.md](docs/glossary.md) — consult it before adding or renaming a feature column.

## Tests

Regression tests (`test_3d_thinning_regression.py`, `test_3d_skimage_comparison.py`, marked `slow`) compare against baselines in `tests/skeletons/*.npz` and `tests/features/*.csv`, generated from a scikit-image brain volume (via `pooch`) — no large dataset is checked into this repo. Use `--update-baseline` deliberately — it overwrites the checked-in ground truth, so only do it when a change is an intentional algorithm update, and inspect the resulting diff before committing.

The equivalent regression tests against the 45-image HRF dataset live in `maskel-evaluations`, since they depend on that external (107MB) dataset rather than anything self-contained.
