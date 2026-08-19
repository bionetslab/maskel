# VesSkel

[![DOI](https://zenodo.org/badge/1198768258.svg)](https://doi.org/10.5281/zenodo.21550587)
[![PyPI version](https://img.shields.io/pypi/v/vesskel.svg)](https://pypi.org/project/vesskel/)
[![Python version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Vessel Skeletonization and Graph-Based Phenotype Analysis in Retinal Fundus Images

## Demo

<video src="https://github.com/user-attachments/assets/8c0febe4-5428-451e-847b-e9210ec0f06b" controls width="1920"></video>

A quick look at what VesSkel can do: initialize a config file with the CLI, load it in the napari plugin to analyze a single example image, then batch-process the whole HRF dataset with the CLI. In practice you'd tweak the extraction settings in napari and save the config back out before batch-processing.

## Installation

```sh
uv sync                  # core only
uv sync --extra dev      # + test tools
uv sync --extra napari   # + napari GUI
uv sync --all-extras     # everything
```

## Napari

```sh
uv sync --extra napari && napari
```

Open a `manual1` TIFF from the HRF folder, then run **Lee94 Thinning** from the VesSkel plugin menu to see the skeleton.

Inside the **Analyze Vessels** widget, tune extraction settings and use **Save Config** to export a reusable JSON preset.

## CLI

Use the same JSON preset exported from napari to batch-process images.

```sh
vesskel init config.json
vesskel validate config.json
vesskel run --input HRF/manual1 --config config.json --out outputs
```

CLI outputs:

- `outputs/summary.csv` with one feature row per image
- Optional per-image skeleton outputs (default: `.npy`)
- Optional per-image branch tables when `output.write_branch_csv=true`
- Optional per-image node tables when `output.write_node_csv=true`
- Optional per-image skeleton graphs when `output.write_graphml=true`

## Configuration

Extraction and output settings are defined in a JSON config file (e.g. the one exported from napari or written by hand).

```json
{
  "schema_version": 3,
  "extraction": {
    "branches": false,
    "branch_color_property": "tortuosity",
    "branch_text": false,
    "nodes": false,
    "summary": true,
    "fractal_dimension": false,
    "vessel_radius": false,
    "junction_cleanup": false,
    "cleanup_threshold_factor": 2.5,
    "prune_spurs": false,
    "min_spur_length": 10.0,
    "spur_iterations": 1,
    "closing_iterations": 0,
    "fill_holes": false,
    "max_hole_size": 0,
    "show_preprocessed": false
  },
  "output": {
    "write_skeleton_npy": true,
    "write_skeleton_png": false,
    "write_summary_csv": true,
    "write_branch_csv": false,
    "write_node_csv": false,
    "write_radius": false,
    "write_graphml": false
  }
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `extraction.branches` | bool | `false` | Extract per-branch features for CSV export or napari visualization |
| `extraction.branch_color_property` | str | `"tortuosity"` | Branch property used to color the napari shapes layer; one of `tortuosity`, `straightness`, `mean_radius`, `std_radius`, `volume`, `surface_area`, ... |
| `extraction.branch_text` | bool | `false` | Display branch ID, length, and tortuosity labels on the napari branch layer |
| `extraction.nodes` | bool | `false` | Extract per-node features for CSV export or napari visualization |
| `extraction.summary` | bool | `true` | Compute summary features  |
| `extraction.fractal_dimension` | bool | `false` | Compute fractal dimension of the skeleton |
| `extraction.vessel_radius` | bool | `false` | Estimate vessel radius using EDT from the segmentation |
| `extraction.junction_cleanup` | bool | `false` | Clean up ambiguous junction pixels after thinning |
| `extraction.cleanup_threshold_factor` | float | `2.5` | Sensitivity for junction cleanup (higher = larger cycles get collapsed) |
| `extraction.prune_spurs` | bool | `false` | Remove short endpoint-to-junction branches (thinning spur artifacts) after skeletonization |
| `extraction.min_spur_length` | float | `10.0` | Branches shorter than this (in pixels) qualify as spurs when `prune_spurs` is true |
| `extraction.spur_iterations` | int | `1` | How often pruning is repeated on its own output, since removing a spur can expose new ones |
| `extraction.closing_iterations` | int | `0` | Morphological closing iterations applied before thinning (0 = disabled) |
| `extraction.fill_holes` | bool | `false` | Fill holes in the binary segmentation before thinning |
| `extraction.max_hole_size` | int | `0` | Maximum hole area (px) to fill when `fill_holes` is true; 0 = fill all |
| `extraction.show_preprocessed` | bool | `false` | Show preprocessed binary layer (after closing and hole filling) in the napari viewer |
| `output.write_skeleton_npy` | bool | `true` | Save skeleton as `.npy` (NumPy array) per image |
| `output.write_skeleton_png` | bool | `false` | Save binary skeleton mask as `.png` per image |
| `output.write_summary_csv` | bool | `true` | Write aggregated per-image features to `summary.csv` |
| `output.write_branch_csv` | bool | `false` | Write per-branch CSV tables (requires `extraction.branches`) |
| `output.write_node_csv` | bool | `false` | Write per-node CSV tables (requires `extraction.nodes`) |
| `output.write_radius` | bool | `false` | Write per-pixel radius matrix as `.npy` (requires `extraction.vessel_radius`) |
| `output.write_graphml` | bool | `false` | Write skeleton graph as `.graphml` per image (nodes = graph nodes, edges = branches) |

### Shell completions

```sh
# zsh
eval "$(vesskel completions zsh)"

# bash
eval "$(vesskel completions bash)"

# PowerShell
vesskel completions powershell | Out-String | Invoke-Expression
```

Add the appropriate line to your shell rc for persistent tab-completion.

## Tests

```sh
uv sync --extra dev && pytest                     # all tests
uv sync --extra dev && pytest -m "not slow"       # skip regression tests
```

- **2D regression** - thinning + feature extraction on all 45 HRF samples, compared against saved baselines
- **3D regression** - thinning + features on a brain volume (from scikit-image), same baseline approach
- **3D comparison** - vesskel `lee94_thin` vs `skimage.morphology.skeletonize` on the brain volume, asserting identical output

First run (or `--update-baseline`) generates baselines in `tests/skeletons/` and `tests/features/`.

## Dataset

This project uses the High-Resolution Fundus (HRF) Image Database, established by a collaborative research group to support comparative studies on automatic segmentation algorithms on retinal fundus images.

The database contains 45 images total:

- 15 images of healthy patients
- 15 images of patients with diabetic retinopathy
- 15 images of glaucomatous patients

Binary gold standard vessel segmentation images and field of view (FOV) masks are available for each image.

### License

> Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.
> Robust Vessel Segmentation in Fundus Images.
> International Journal of Biomedical Imaging, vol. 2013, 2013

The HRF dataset is released under the **Creative Commons 4.0 Attribution License**.

For more information, visit the [HRF Image Database](https://www5.cs.fau.de/research/data/fundus-images/).

## Citation

If you use VesSkel in your research, please cite the Zenodo release:

> Wittmann, S. (2026). 404Simon/VesSkel: 5.1.0 (Version 5.1.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.21550587>

## License

VesSkel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
