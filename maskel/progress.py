"""A one-line progress bar with an ETA."""

from __future__ import annotations

import os
import sys
import time

from maskel.tiling import Tile, TileGrid

PROGRESS_STYLES = ("off", "auto", "bar", "lines")
PROGRESS_ENV_VAR = "MASKEL_PROGRESS"

_BAR_WIDTH = 24
_LOG_STEP_PERCENT = 5


def format_duration(seconds: float) -> str:
    """``45s``, ``3m12s``, ``2h07m``."""
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
    """Resolve a requested style to ``off``, ``bar`` or ``lines``.

    ``auto`` picks ``bar`` where a carriage return renders: a terminal,
    PyCharm's console, or anywhere `PROGRESS_ENV_VAR` says so.
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
    """Block characters, or ASCII where the stream cannot encode them."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "█░".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "#", "-"
    return "█", "░"


class ProgressReporter:
    """Tile progress weighted by voxels, since edge tiles are much smaller.

    ``bar`` rewrites one line in place; ``lines`` prints a new line every few
    percent, for logs.
    """

    def __init__(self, label: str, grid: TileGrid, progress: str | bool) -> None:
        self.label = label
        self.total_items = len(grid)
        self.total_units = max(sum(tile.read_size for tile in grid), 1)
        self.done_items = 0
        self.done_units = 0
        self.start = time.monotonic()
        self.stream = sys.stdout
        self.style = resolve_progress_style(progress, self.stream)
        self.enabled = self.style != "off"
        self.filled_glyph, self.empty_glyph = bar_glyphs(self.stream)
        self._painted_width = 0
        self._logged_step = -1
        if self.enabled:
            self._paint()

    def advance(self, tile: Tile) -> None:
        """Count one finished tile."""
        self.done_items += 1
        self.done_units += tile.read_size
        if self.enabled:
            self._paint()

    def done(self) -> None:
        """Close the bar and print the total time."""
        if not self.enabled:
            return
        if self.style == "bar":
            self.stream.write("\n")
        elapsed = format_duration(time.monotonic() - self.start)
        self.stream.write(f"{self.label} done in {elapsed}.\n")
        self.stream.flush()

    def _line(self) -> str:
        fraction = self.done_units / self.total_units
        elapsed = time.monotonic() - self.start
        remaining = self.total_units - self.done_units
        eta = (
            format_duration(elapsed / self.done_units * remaining)
            if self.done_units
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
            # Pad to the widest line so far so no tail of an earlier one shows.
            self.stream.write("\r" + line.ljust(self._painted_width))
            self._painted_width = max(self._painted_width, len(line))
            self.stream.flush()
            return

        # One line per _LOG_STEP_PERCENT, plus the last tile.
        step = int(self.done_units * 100 // self.total_units) // _LOG_STEP_PERCENT
        if step == self._logged_step and self.done_items != self.total_items:
            return
        self._logged_step = step
        self.stream.write(line + "\n")
        self.stream.flush()
