"""A single-line progress bar with an ETA, for runs measured in hours."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable
from math import prod

from vesskel.tiling.grid import TileGrid, TileSpec

PROGRESS_STYLES = ("off", "auto", "bar", "lines")
PROGRESS_ENV_VAR = "VESSKEL_PROGRESS"

_BAR_WIDTH = 24

# Emit a fresh log line only every this many percent in "lines" mode, so a
# redirected run produces ~20 lines instead of one per tile.
_LOG_STEP_PERCENT = 5


def format_duration(seconds: float) -> str:
    """Compact, human-scannable duration: ``45s``, ``3m12s``, ``2h07m``."""
    if seconds < 1:
        return "<1s"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def resolve_progress_style(style: str | bool, stream) -> str:
    """Turn a requested style into one of ``off`` / ``bar`` / ``lines``.

    ``auto`` picks the in-place bar when something can interpret a carriage
    return. ``isatty()`` alone is too strict: PyCharm's run console and many
    CI viewers are pipes that still render ``\\r`` correctly, so an explicit
    override is available - both as `PROGRESS_ENV_VAR` and, from the CLI,
    ``--progress``.
    """
    if style is True:
        style = "auto"
    elif style is False:
        style = "off"
    if style not in PROGRESS_STYLES:
        raise ValueError(f"progress must be one of {PROGRESS_STYLES}, got {style!r}")
    if style != "auto":
        return style

    override = os.environ.get(PROGRESS_ENV_VAR)
    if override in PROGRESS_STYLES and override != "auto":
        return override
    if getattr(stream, "isatty", bool)():
        return "bar"
    if os.environ.get("PYCHARM_HOSTED") == "1":
        return "bar"
    return "lines"


def bar_glyphs(stream) -> tuple[str, str]:
    """Block characters where the stream can encode them, ASCII otherwise.

    Redirecting to a file on Windows gives a cp1252 stream, and writing a
    block character to that raises UnicodeEncodeError - a progress bar must
    never be the thing that kills a six-hour run.
    """
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "█░".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "#", "-"
    return "█", "░"


class ProgressReporter:
    """Tile progress, weighted by voxels rather than by tile count.

    Edge tiles can be a small fraction of an interior one - on a grid whose
    last tile is 2 voxels deep, counting tiles would report a percentage that
    bears no relation to the work left. Voxels track the real cost far better.

    The estimate is still only an extrapolation of the mean rate: the first
    tile carries numba's JIT compilation, and thinning time depends on how
    much foreground a tile holds, so early ETAs run high and settle as more
    tiles land.

    In ``bar`` style the line rewrites itself in place with ``\\r``. In
    ``lines`` style - for logs, where there is no cursor to move - it emits a
    newline-terminated line every few percent instead.
    """

    def __init__(self, label: str, grid: TileGrid, progress: str | bool) -> None:
        self.label = label
        self.total_items = len(grid)
        self.total_units = max(grid.read_total, 1)
        self.done_items = 0
        self.done_units = 0
        # Units this run actually computed. Tiles inherited from an
        # interrupted run count toward the bar but not toward the rate: they
        # took no time *now*, and folding them in would make the throughput
        # look enormous and the ETA absurdly short.
        self.measured_units = 0
        self.start = time.monotonic()
        # Resolved now rather than at import: pytest and friends swap stdout.
        self.stream = sys.stdout
        self.style = resolve_progress_style(progress, self.stream)
        self.enabled = self.style != "off"
        self.filled_glyph, self.empty_glyph = bar_glyphs(self.stream)
        self._painted_width = 0
        self._logged_step = -1
        if self.enabled:
            self._paint()

    def advance(self, spec: TileSpec) -> None:
        self.done_items += 1
        self.done_units += prod(spec.read_shape)
        self.measured_units += prod(spec.read_shape)
        if self.enabled:
            self._paint()

    def skip(self, specs: Iterable[TileSpec]) -> None:
        """Credit tiles a previous run already finished, without timing them."""
        skipped = 0
        for spec in specs:
            self.done_items += 1
            self.done_units += prod(spec.read_shape)
            skipped += 1
        if skipped and self.enabled:
            self._paint()

    def done(self) -> None:
        if not self.enabled:
            return
        if self.style == "bar":
            # Close off the line the bar has been overwriting.
            self.stream.write("\n")
        elapsed = format_duration(time.monotonic() - self.start)
        self.stream.write(f"{self.label} done in {elapsed}.\n")
        self.stream.flush()

    # -- rendering ---------------------------------------------------------

    def _line(self) -> str:
        fraction = self.done_units / self.total_units
        elapsed = time.monotonic() - self.start
        remaining = self.total_units - self.done_units
        # Rate comes from measured work only, so a resumed run predicts from
        # what it has actually done rather than from what it inherited.
        eta = (
            format_duration(elapsed / self.measured_units * remaining)
            if self.measured_units
            else "?"
        )
        filled = int(_BAR_WIDTH * fraction)
        bar = self.filled_glyph * filled + self.empty_glyph * (_BAR_WIDTH - filled)
        return (
            f"{self.label} [{bar}] {fraction:5.1%}  "
            f"{self.done_items}/{self.total_items} tiles  "
            f"elapsed {format_duration(elapsed)}  ETA {eta}"
        )

    def _paint(self) -> None:
        line = self._line()
        if self.style == "bar":
            # Pad to the widest line written so far, so a shorter repaint
            # cannot leave a tail of an earlier one behind. Tracking the max
            # rather than the last width matters because the line can shrink
            # twice running (e.g. an ETA falling 10s -> 9s -> 8s).
            self.stream.write("\r" + line.ljust(self._painted_width))
            self._painted_width = max(self._painted_width, len(line))
            self.stream.flush()
            return

        step = int(self.done_units * 100 // self.total_units) // _LOG_STEP_PERCENT
        if step == self._logged_step and self.done_items != self.total_items:
            return
        self._logged_step = step
        self.stream.write(line + "\n")
        self.stream.flush()
