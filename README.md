# Maskel

[![PyPI version](https://img.shields.io/pypi/v/maskel.svg)](https://pypi.org/project/maskel/)
[![Python version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Skeletonization and graph-based feature extraction for branching biological structures — vasculature, fibers, neurites, and other network-like objects — from 2D or 3D binary or multi-label segmentation masks.

Maskel is the core algorithm package: thinning, feature extraction, and the batch CLI. For the napari plugin, see [napari-maskel](https://github.com/bionetslab/napari-maskel). For benchmarks and the HRF-based analysis notebooks, see [maskel-evaluations](https://github.com/bionetslab/maskel-evaluations).

**Full documentation: https://bionetslab.github.io/maskel/**

## Installation

```sh
uv sync                  # core only
uv sync --extra dev      # + test tools
```

## Quick start

```sh
maskel init config.json
maskel validate config.json
maskel run --input /path/to/images --config config.json --out outputs
```

See the [full documentation](https://bionetslab.github.io/maskel/) for the complete config schema and CLI options.

## Tests

```sh
uv sync --extra dev && pytest                     # all tests
uv sync --extra dev && pytest -m "not slow"       # skip the slow 3D comparison test
```

## License

Maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
