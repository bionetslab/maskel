Skeletonization and graph-based feature extraction for branching biological structures — vasculature, fibers, neurites, and other network-like objects — from 2D or 3D binary or multi-label segmentation masks.

Maskel is the core algorithm package: thinning, feature extraction, and the batch CLI. It has no napari dependency — for the interactive napari plugin, see **[napari-maskel](https://bionetslab.github.io/napari-maskel/)**. For benchmarks and the HRF-based analysis notebooks, see [maskel-evaluations](https://github.com/bionetslab/maskel-evaluations) on GitHub.

<video controls style="max-width: 100%;">
  <source src="demo.mp4" type="video/mp4">
</video>

## What it does

Given a 2D or 3D segmentation mask — either a plain binary mask or a multi-object instance segmentation map — from any imaging modality or biological structure, maskel:

1. Optionally preprocesses the binary mask (morphological closing, hole filling)
2. Thins it to a 1-pixel/voxel skeleton (a fast, Numba-parallelized implementation of Lee et al. 1994 thinning)
3. Builds a graph over the skeleton (nodes = junctions/endpoints, edges = branches)
4. Optionally prunes short spur branches and cleans up ambiguous triangle junctions
5. Extracts per-branch, per-node, and per-object summary features (length, tortuosity, radius, fractal dimension, ...)

Touching-but-distinct objects in a multi-object mask are always skeletonized independently, never merged.

See [Installation](installation.md) to get started, [CLI Usage](cli.md) for the batch command line, and the [Configuration Reference](configuration.md) for every extraction/output option.

## License

Maskel is released under the **MIT License**. See [LICENSE](https://github.com/bionetslab/maskel/blob/main/LICENSE) for details.
