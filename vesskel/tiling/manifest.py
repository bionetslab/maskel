"""On-disk bookkeeping: the tile manifest and the done log.

Layout of a work directory::

    work_dir/
      manifest.json           grid parameters + per-tile coordinates
      done.jsonl              one line per finished tile, appended as it lands
      <base_name>_skeleton.npy    the result (`SKELETON_NAME` by default)

Nothing else is written - no per-tile staging files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType

from vesskel.tiling.grid import TileGrid, TileSpec

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
    """Append-only record of finished tiles, for crash resume.

    One JSON object per line holding nothing but the tile index - that is all
    a resume needs, and a fixed tiny record keeps the append atomic enough
    that a kill mid-write costs at most the one tile whose line was cut.

    Used as a context manager while a run is in progress; `read` and `clear`
    work without opening it.
    """

    def __init__(self, work_dir: Path) -> None:
        self.path = Path(work_dir) / DONE_LOG_NAME
        self._handle = None

    def read(self) -> set[tuple[int, ...]]:
        """Indices of tiles a previous run finished.

        A truncated final line (the process was killed mid-write) is skipped
        rather than fatal - that tile simply gets redone.
        """
        if not self.path.is_file():
            return set()
        done: set[tuple[int, ...]] = set()
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(tuple(json.loads(line)["index"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return done

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def record(self, spec: TileSpec) -> None:
        """Append *spec* and flush, so the line survives a kill."""
        if self._handle is None:
            raise RuntimeError("DoneLog.record() outside of a `with` block")
        self._handle.write(json.dumps({"index": list(spec.index)}) + "\n")
        self._handle.flush()

    def __enter__(self) -> DoneLog:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()
