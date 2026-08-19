"""Internal utility functions shared across the maskel package."""

import numpy as np


def to_binary(arr: np.ndarray) -> np.ndarray:
    """Convert an array to a binary uint8 mask (0=background, >0=foreground)."""
    return (arr > 0).astype(np.uint8)
