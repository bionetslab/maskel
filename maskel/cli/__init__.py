"""Standalone CLI for Maskel batch analysis.

Each subcommand lives in its own module, which registers its arguments via
``add_parser`` and its handler via ``set_defaults(handler=...)``:

- `maskel.cli.run` - ``run``
- `maskel.cli.tiled` - ``skeletonize_tiled``
- `maskel.cli.config` - ``init``, ``validate``
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version

from maskel.cli import config, run, tiled


def _completions(args: argparse.Namespace) -> int:
    from argcomplete import shellcode

    print(shellcode(["maskel"], shell=args.shell))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskel",
        description="Maskel CLI for batch mask analysis.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"maskel {version('maskel')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run.add_parser(subparsers)
    tiled.add_parser(subparsers)
    config.add_parsers(subparsers)

    completions_parser = subparsers.add_parser(
        "completions",
        help="Print shell completion script to stdout.",
    )
    completions_parser.add_argument(
        "shell",
        choices=("bash", "zsh", "powershell"),
        help="Target shell.",
    )
    completions_parser.set_defaults(handler=_completions)

    return parser


# WARNING: keep this guard, everything above it and the top-level imports of
# the subcommand modules free of heavy imports (numpy, PIL, maskel.pipeline,
# maskel._io). The guard lets shell completions exit in 81 ms instead of
# 800 ms. Tests in tests/test_cli.py::TestCompletionSpeed enforce this.
if "_ARGCOMPLETE" in os.environ or (len(sys.argv) > 1 and sys.argv[1] == "completions"):
    if "_ARGCOMPLETE" in os.environ:
        try:
            from argcomplete import autocomplete

            autocomplete(make_parser())
        except ImportError:
            pass
    else:
        try:
            from argcomplete import shellcode

            shell = sys.argv[2] if len(sys.argv) > 2 else "zsh"
            print(shellcode(["maskel"], shell=shell))
        except ImportError:
            pass
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    parser = make_parser()
    try:
        from argcomplete import autocomplete

        autocomplete(parser)
    except ImportError:
        pass
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return args.handler(args)
