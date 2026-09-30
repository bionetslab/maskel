"""What a run writes into its work directory.

- ``manifest.json`` - grid parameters and per-tile coordinates
- ``done.jsonl`` - one line per finished tile
- ``<base_name>_skeleton.npy`` - the result
"""

from __future__ import annotations

import json
from pathlib import Path

from maskel.tiling.grid import TileGrid, TileSpec

MANIFEST_NAME = "manifest.json"
DONE_LOG_NAME = "done.jsonl"
SKELETON_NAME = "skeleton.npy"


def write_manifest(grid: TileGrid, work_dir: Path) -> Path:
    """Write *grid* to ``work_dir/manifest.json``, creating the directory."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / MANIFEST_NAME
    with path.open("w") as f:
        json.dump(grid.to_dict(), f, indent=2)
    return path


def read_manifest(work_dir: Path) -> TileGrid:
    """Rebuild the `TileGrid` recorded in ``work_dir/manifest.json``."""
    with (Path(work_dir) / MANIFEST_NAME).open() as f:
        return TileGrid.from_dict(json.load(f))


class DoneLog:
    """Append-only list of finished tile indices, used to resume a crashed run."""

    def __init__(self, work_dir: Path) -> None:
        self.path = Path(work_dir) / DONE_LOG_NAME

    def read(self) -> set[tuple[int, ...]]:
        """Indices of finished tiles. A truncated last line is skipped."""
        if not self.path.is_file():
            return set()
        done: set[tuple[int, ...]] = set()
        with self.path.open() as f:
            for line in f:
                try:
                    done.add(tuple(json.loads(line)["index"]))
                except (ValueError, KeyError, TypeError):
                    continue
        return done

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def record(self, spec: TileSpec) -> None:
        """Append one tile. Opened per call so the line is on disk immediately."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"index": list(spec.index)}) + "\n")
