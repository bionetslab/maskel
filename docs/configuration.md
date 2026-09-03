# Configuration Reference

Extraction and output settings are defined in a JSON config file (e.g. the one exported from napari or written by hand). This is the schema shared by the maskel CLI and the [napari-maskel](https://bionetslab.github.io/napari-maskel/) plugin.

```json
{
  "schema_version": 6,
  "extraction": {
    "branches": false,
    "branch_color_property": "tortuosity",
    "branch_text": false,
    "nodes": false,
    "summary": true,
    "fractal_dimension": false,
    "mask_radius": false,
    "junction_cleanup": false,
    "cleanup_threshold_factor": 2.5,
    "prune_spurs": false,
    "min_spur_length": 10.0,
    "spur_iterations": 1,
    "closing_iterations": 0,
    "fill_holes": false,
    "max_hole_size": 0,
    "show_preprocessed": false,
    "spacing": null
  },
  "output": {
    "write_skeleton_npy": true,
    "write_skeleton_png": false,
    "write_summary_csv": true,
    "write_branch_csv": false,
    "write_node_csv": false,
    "write_radius": false,
    "write_graphml": false,
    "write_networkx_graph": false
  }
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `extraction.branches` | bool | `false` | Extract per-branch features for CSV export or napari visualization |
| `extraction.branch_color_property` | str | `"tortuosity"` | Branch property used to color the napari shapes layer; one of `object_id`, `tortuosity`, `straightness`, `mean_radius`, `std_radius`, `volume`, `surface_area`, ... |
| `extraction.branch_text` | bool | `false` | Display branch ID, length, and tortuosity labels on the napari branch layer |
| `extraction.nodes` | bool | `false` | Extract per-node features for CSV export or napari visualization |
| `extraction.summary` | bool | `true` | Compute summary features |
| `extraction.fractal_dimension` | bool | `false` | Compute fractal dimension of the skeleton |
| `extraction.mask_radius` | bool | `false` | Estimate mask radius using EDT from the segmentation |
| `extraction.junction_cleanup` | bool | `false` | Clean up ambiguous junction pixels after thinning |
| `extraction.cleanup_threshold_factor` | float | `2.5` | Sensitivity for junction cleanup (higher = larger cycles get collapsed) |
| `extraction.prune_spurs` | bool | `false` | Remove short endpoint-to-junction branches (thinning spur artifacts) after skeletonization |
| `extraction.min_spur_length` | float | `10.0` | Branches shorter than this qualify as spurs when `prune_spurs` is true; in pixel units, or physical units when `spacing` is set |
| `extraction.spur_iterations` | int | `1` | How often pruning is repeated on its own output, since removing a spur can expose new ones |
| `extraction.closing_iterations` | int | `0` | Morphological closing iterations applied before thinning (0 = disabled) |
| `extraction.fill_holes` | bool | `false` | Fill holes in the binary segmentation before thinning |
| `extraction.max_hole_size` | int | `0` | Maximum hole area (px) to fill when `fill_holes` is true; 0 = fill all |
| `extraction.show_preprocessed` | bool | `false` | Show preprocessed binary layer (after closing and hole filling) in the napari viewer |
| `extraction.spacing` | list[float] or `null` | `null` | Per-axis physical pixel/voxel size (length must match the image's dimensionality: 2 for 2D, 3 for 3D). When set, length/area/volume features come out in physical units instead of pixel units. A per-image dimensionality mismatch falls back to `null` (isotropic pixel units) with a warning rather than failing the batch. Box-counting `fractal_dimension` is only valid for isotropic voxels, so it's forced to `0.0` (with a warning) whenever spacing is set and anisotropic. |
| `output.write_skeleton_npy` | bool | `true` | Save skeleton as `.npy` (NumPy array) per image |
| `output.write_skeleton_png` | bool | `false` | Save binary skeleton mask as `.png` per image |
| `output.write_summary_csv` | bool | `true` | Write aggregated per-image features to `summary.csv` |
| `output.write_branch_csv` | bool | `false` | Write per-branch CSV tables (requires `extraction.branches`) |
| `output.write_node_csv` | bool | `false` | Write per-node CSV tables (requires `extraction.nodes`) |
| `output.write_radius` | bool | `false` | Write per-pixel radius matrix as `.npy` (requires `extraction.mask_radius`) |
| `output.write_graphml` | bool | `false` | Write skeleton graph as `.graphml` per image (nodes = graph nodes, edges = branches) |
| `output.write_networkx_graph` | bool | `false` | Write skeleton graph as a pickled `networkx.MultiGraph` per image — same node/edge/graph attributes as `write_graphml`, but keeps NaN values and native Python types rather than GraphML/yEd-safe ones |

Unknown keys warn rather than error, so configs written against an older schema version stay loadable across additive changes.
