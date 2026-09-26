"""`dat-distiller` command line: the app serves itself, no separate web server."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from .api.app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8756


def serve(args: argparse.Namespace) -> int:
    """Run the API, and the built frontend when there is one, on localhost."""
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dat-distiller",
        description="Generate synthetic data, label it with Jev, and train models.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve_parser = subcommands.add_parser(
        "serve",
        help="Serve the API and the built frontend on localhost.",
    )
    serve_parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="address to listen on (default: %(default)s).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="port to listen on (default: %(default)s).",
    )
    serve_parser.set_defaults(handler=serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
