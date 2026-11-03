"""`maskel init` and `maskel validate`: write and check config JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maskel.config import PipelineConfig, load_pipeline_config


def add_parsers(subparsers: argparse._SubParsersAction) -> None:
    init_parser = subparsers.add_parser(
        "init",
        help="Create a starter config JSON.",
    )
    init_parser.add_argument(
        "out", nargs="?", default="maskel.json", help="Output config path."
    )
    init_parser.set_defaults(handler=config_init)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate and print a normalised config JSON.",
    )
    validate_parser.add_argument(
        "config", nargs="?", default="maskel.json", help="Config JSON path."
    )
    validate_parser.set_defaults(handler=validate_config)


def config_init(args: argparse.Namespace) -> int:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(PipelineConfig().to_dict(), f, indent=2)
    print(f"Wrote starter config to '{path}'.")
    return 0


def validate_config(args: argparse.Namespace) -> int:
    config = load_pipeline_config(Path(args.config))
    print(json.dumps(config.to_dict(), indent=2))
    print("Configuration is valid.")
    return 0
