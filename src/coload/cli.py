"""Command-line interface.

Server side: ``coload serve``. Client side (works from Windows, WSL, or any
box that can reach the gateway): ``coload models``, ``coload up <model>``,
``coload down <model>``, ``coload chat <model> <prompt>``, ``coload status``.

The gateway URL comes from ``--url``, else the ``COLOAD_URL`` env var, else
``http://127.0.0.1:8800``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import httpx

from . import __version__

DEFAULT_URL = "http://127.0.0.1:8800"


class GatewayError(RuntimeError):
    """A gateway request failed; the message is ready for the user."""


class GatewayClient:
    """Small synchronous HTTP client for the coload gateway."""

    def __init__(self, base_url: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(600.0, connect=5.0),
        )
        self._base_url = base_url.rstrip("/")

    def status(self) -> dict:
        return self._json(self._request("GET", "/status"))

    def models(self) -> list[dict]:
        """Configured models merged with live residency."""
        catalog = self._json(self._request("GET", "/v1/models"))["data"]
        status = self.status()
        resident: set[str] = set()
        for engine in status.get("engines", {}).values():
            resident.update(engine.get("resident", []))
        return [
            {
                "model": entry["id"],
                "engine": entry["owned_by"],
                # Ollama reports name:tag; a bare name matches its :latest
                "resident": entry["id"] in resident
                or f"{entry['id']}:latest" in resident,
            }
            for entry in catalog
        ]

    def load(self, model: str) -> dict:
        return self._json(
            self._request("POST", "/models/load", json={"model": model})
        )

    def unload(self, model: str) -> dict:
        return self._json(
            self._request("POST", "/models/unload", json={"model": model})
        )

    def chat(self, model: str, prompt: str) -> str:
        resp = self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return self._json(resp)["choices"][0]["message"]["content"]

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise GatewayError(
                f"could not reach coload gateway at {self._base_url}: {exc}\n"
                f"Is it running? Start it with: coload serve"
            ) from exc

    def _json(self, resp: httpx.Response) -> dict:
        payload = resp.json()
        if resp.status_code >= 400:
            message = payload.get("error", {}).get("message", resp.text)
            raise GatewayError(message)
        return payload


def format_models_table(rows: list[dict]) -> str:
    if not rows:
        return "no models configured"
    width_model = max(len("MODEL"), *(len(r["model"]) for r in rows))
    width_engine = max(len("ENGINE"), *(len(r["engine"]) for r in rows))
    lines = [f"{'MODEL':<{width_model}}  {'ENGINE':<{width_engine}}  RESIDENT"]
    for r in rows:
        mark = "yes" if r["resident"] else "-"
        lines.append(f"{r['model']:<{width_model}}  {r['engine']:<{width_engine}}  {mark}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coload",
        description="Co-load multiple inference engines onto one GPU.",
    )
    parser.add_argument("--version", action="version", version=f"coload {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_url(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--url",
            default=os.environ.get("COLOAD_URL", DEFAULT_URL),
            help="gateway base URL (or set COLOAD_URL)",
        )

    serve = sub.add_parser("serve", help="run the gateway + orchestrator")
    serve.add_argument(
        "-c",
        "--config",
        default=None,
        help="config file (default: config.local.yaml if present, else config.yaml)",
    )
    serve.add_argument("--host", default=None, help="override config host")
    serve.add_argument("--port", type=int, default=None, help="override config port")

    for name, help_text in [
        ("status", "show the VRAM map and what's resident"),
        ("models", "list configured models and residency"),
    ]:
        add_url(sub.add_parser(name, help=help_text))

    up = sub.add_parser("up", help="spin a model up now (warm it)")
    up.add_argument("model")
    add_url(up)

    down = sub.add_parser("down", help="unload a model, freeing its VRAM")
    down.add_argument("model")
    add_url(down)

    chat = sub.add_parser("chat", help="one-shot prompt from the command line")
    chat.add_argument("model")
    chat.add_argument("prompt", nargs="+", help="the prompt text")
    add_url(chat)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args)

    gw = GatewayClient(args.url)
    try:
        if args.command == "status":
            print(json.dumps(gw.status(), indent=2))
        elif args.command == "models":
            print(format_models_table(gw.models()))
        elif args.command == "up":
            result = gw.load(args.model)
            print(f"{result['model']}: {result['status']}")
        elif args.command == "down":
            result = gw.unload(args.model)
            print(f"{result['model']}: {result['status']}")
        elif args.command == "chat":
            print(gw.chat(args.model, " ".join(args.prompt)))
    except GatewayError as exc:
        print(f"coload: {exc}", file=sys.stderr)
        return 1
    return 0


def _serve(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from .config import load_config, resolve_config_path
    from .runtime import build_runtime

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config_path = resolve_config_path(args.config)
    logging.getLogger("coload").info("using config: %s", config_path)
    config = load_config(config_path)
    runtime = build_runtime(config)
    uvicorn.run(
        runtime.app,
        host=args.host or config.host,
        port=args.port or config.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
