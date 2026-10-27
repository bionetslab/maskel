"""Per-block preprocessing: opening, closing and bounded hole filling.

All of them are local enough to run tile by tile, as long as each block is read
with `BlockPreprocess.margin` voxels on top of whatever the thinning needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maskel.pipeline import preprocess_binary


@dataclass(frozen=True)
class BlockPreprocess:
    """Preprocessing to run on each block before thinning."""

    opening_radius: int = 0
    closing_iterations: int = 0
    fill_holes: bool = False
    max_hole_size: int = 0

    def __post_init__(self) -> None:
        if self.opening_radius < 0:
            raise ValueError(
                f"opening_radius must be non-negative, got {self.opening_radius}"
            )
        if self.closing_iterations < 0:
            raise ValueError(
                f"closing_iterations must be non-negative, "
                f"got {self.closing_iterations}"
            )
        if self.max_hole_size < 0:
            raise ValueError(
                f"max_hole_size must be non-negative, got {self.max_hole_size}"
            )
        if self.fill_holes and self.max_hole_size < 1:
            raise ValueError(
                "Tiled fill_holes needs max_hole_size > 0. An unbounded hole can "
                "span the whole volume, so no halo is wide enough to catch it."
            )

    def __bool__(self) -> bool:
        """False when there is nothing to do."""
        return self.opening_radius > 0 or self.closing_iterations > 0 or self.fill_holes

    @property
    def margin(self) -> int:
        """Halo these steps need on top of the thinning halo.

        Each step reads a neighbourhood of its own input, so the reaches of
        consecutive steps add up. Voxels near the edge of a block are padded
        (edge-repeated) rather than read, which makes them wrong there; the
        error travels inward by at most each step's reach:

        - Opening and closing with a ball of radius r: erosion plus dilation,
          r each, so 2r per step.
        - Hole filling: a hole of V voxels spans at most V along an axis, so
          max_hole_size covers every hole that is kept - a wider one gets
          reverted by the size filter anyway.
        """
        holes = self.max_hole_size if self.fill_holes else 0
        return 2 * self.opening_radius + 2 * self.closing_iterations + holes

    def apply(self, block: np.ndarray) -> np.ndarray:
        """Preprocess *block*, in the same order as the whole-volume path."""
        return preprocess_binary(
            block,
            closing_iterations=self.closing_iterations,
            fill_holes=self.fill_holes,
            max_hole_size=self.max_hole_size,
            opening_radius=self.opening_radius,
        )

    def to_dict(self) -> dict[str, object]:
        """Manifest record, used to reject a resume with different settings."""
        return {
            "opening_radius": self.opening_radius,
            "closing_iterations": self.closing_iterations,
            "fill_holes": self.fill_holes,
            "max_hole_size": self.max_hole_size,
        }
