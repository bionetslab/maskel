"""Writing feature tables (CSV) and graphs (GraphML, pickle)."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from maskel.config import OutputConfig
from maskel.features import LabelFeatures
from maskel.io.graphml import write_graphml, write_networkx_pickle


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write *rows* to a CSV with the union of their keys as sorted columns; nothing if empty."""
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: v.item() if isinstance(v, np.generic) else v
                    for k, v in row.items()
                }
            )


def write_features(
    out_dir: Path,
    name: str,
    results: Iterable[LabelFeatures],
    output: OutputConfig,
) -> None:
    """Write every label's graphs as they come, then the summary, branch and node tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, branches, nodes = [], [], []
    for result in results:
        if result.summary:
            summary.append({"image": name, **result.summary})
        branches.extend(result.branch_records)
        nodes.extend(result.node_records)
        graph_args = (result.graph, result.branch_data)
        graph_kwargs = {
            "summary_features": result.summary or None,
            "radius": result.radius,
        }
        if output.write_graphml:
            path = out_dir / f"{name}_{result.object_id}_graph.graphml"
            write_graphml(*graph_args, path, **graph_kwargs)
        if output.write_networkx_graph:
            path = out_dir / f"{name}_{result.object_id}_graph.pkl"
            write_networkx_pickle(*graph_args, path, **graph_kwargs)

    if output.write_summary_csv:
        write_csv(out_dir / f"{name}_summary.csv", summary)
    if output.write_branch_csv:
        write_csv(out_dir / f"{name}_branches.csv", branches)
    if output.write_node_csv:
        write_csv(out_dir / f"{name}_nodes.csv", nodes)
