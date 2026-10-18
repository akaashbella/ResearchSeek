"""Command-line entry point."""

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="researchseek",
        description="Local semantic and keyword search over research notes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, summary in (
        ("index", "Index research notes"),
        ("search", "Search indexed passages"),
        ("benchmark", "Evaluate retrieval correctness, speed, and quality"),
    ):
        commands.add_parser(
            name,
            help=f"{summary} (not implemented yet)",
            description=f"{summary}. Not implemented yet.",
        )

    args = parser.parse_args(argv)
    print(f"researchseek: {args.command} is not implemented yet.", file=sys.stderr)
    return 1
