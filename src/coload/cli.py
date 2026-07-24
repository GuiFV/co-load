"""Command-line interface: ``coload serve`` and ``coload status``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import httpx

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coload",
        description="Co-load multiple inference engines onto one GPU.",
    )
    parser.add_argument("--version", action="version", version=f"coload {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the gateway + orchestrator")
    serve.add_argument("-c", "--config", default="config.yaml", help="path to config.yaml")
    serve.add_argument("--host", default=None, help="override config host")
    serve.add_argument("--port", type=int, default=None, help="override config port")

    status = sub.add_parser("status", help="show what's resident and the VRAM map")
    status.add_argument(
        "--url", default="http://127.0.0.1:8800", help="gateway base URL"
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return _status(args)


def _serve(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from .config import load_config
    from .runtime import build_runtime

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    runtime = build_runtime(config)
    uvicorn.run(
        runtime.app,
        host=args.host or config.host,
        port=args.port or config.port,
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    try:
        resp = httpx.get(f"{args.url.rstrip('/')}/status", timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"coload: could not reach gateway at {args.url}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
