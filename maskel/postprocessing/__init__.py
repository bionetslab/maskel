"""Skeleton cleanup after thinning: junction collapsing and spur pruning."""

from maskel.postprocessing.junction_cleanup import collapse_triangle_junctions
from maskel.postprocessing.spur_pruning import prune_short_spurs

__all__ = ["collapse_triangle_junctions", "prune_short_spurs"]
