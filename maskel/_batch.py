"""Batch worker for multiprocessing."""

from __future__ import annotations

from pathlib import Path

from maskel._io import load_image, save_analysis_outputs
from maskel.config import PipelineConfig
from maskel.pipeline import analyze_binary_image


def process_one(
    in_path: Path,
    safe_name: str,
    out_dir: Path,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    """Load, analyse, save one image. Returns one summary row per object for agg CSV."""
    image = load_image(in_path)
    result = analyze_binary_image(image=image, config=config)

    save_analysis_outputs(
        out_dir, safe_name, result, config.output, write_summary=False
    )

    return [{"image": in_path.name, **row} for row in result.summary_features]
