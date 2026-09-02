# CLI Usage

```sh
maskel init config.json
maskel validate config.json
maskel run --input /path/to/images --config config.json --out outputs --jobs 0
```

A config JSON can also be exported from the [napari-maskel](https://bionetslab.github.io/napari-maskel/) plugin's **Save recipe** button and used here directly — both consume the same schema (see [Configuration Reference](configuration.md)).

Input images can be either a plain binary segmentation mask, or a multi-object instance segmentation map (more than one distinct nonzero value). In the latter case, every object is skeletonized independently — touching-but-distinct objects are correctly kept separate rather than merged into one skeleton — and every output row is tagged with the `object_id` it came from (the mask's own label value; `1` for a plain binary mask).

## Outputs

- `outputs/summary.csv` with one feature row per object per image
- Optional per-image skeleton outputs (default: `.npy`)
- Optional per-image branch tables when `output.write_branch_csv=true` (includes an `object_id` column)
- Optional per-image node tables when `output.write_node_csv=true` (includes an `object_id` column)
- Optional per-object skeleton graphs when `output.write_graphml=true` (one `_<object_id>_graph.graphml` file per object)
- Optional per-object pickled networkx graphs when `output.write_networkx_graph=true` (one `_<object_id>_graph.pkl` file per object; richer than GraphML — keeps NaN attributes and native Python types)

## Batch parallelism

`maskel run --jobs N` spawns a process pool over all input images, one process per image; `--jobs 0` uses all CPU cores. `Ctrl-C` triggers a clean worker shutdown rather than hanging.
