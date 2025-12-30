import argparse
import json
import sys
import time
from typing import List, Optional

import httpx


DEFAULT_API_BASE = "http://127.0.0.1:8000"


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _print_status(status: dict) -> None:
    if not status:
        print("No profiling status available.")
        return
    running = status.get("running")
    current = status.get("current")
    completed = status.get("completed", 0)
    total = status.get("total", 0)
    if running:
        suffix = f" ({current})" if current else ""
        print(f"Profiling {completed}/{total}{suffix}")
    else:
        print(f"Profiling complete ({completed}/{total}).")
    errors = status.get("errors") or []
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"- {err.get('model_key')}: {err.get('error')}")


def _poll_status(client: httpx.Client, base: str, timeout_s: int = 600) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        resp = client.get(_join_url(base, "/api/models/profile"), timeout=10)
        resp.raise_for_status()
        status = resp.json().get("status") or {}
        _print_status(status)
        if not status.get("running"):
            return
        time.sleep(2)
    print("Timed out waiting for profiling to finish.")


def run_models_profile(args: argparse.Namespace) -> int:
    base = args.base_url or DEFAULT_API_BASE
    payload = {"force": args.force, "models": args.models or []}
    with httpx.Client() as client:
        resp = client.post(_join_url(base, "/api/models/profile"), json=payload, timeout=10)
        if resp.status_code >= 400:
            print(f"Failed to start profiling: HTTP {resp.status_code}")
            return 1
        data = resp.json()
        _print_status(data.get("status") or {})
        if args.wait:
            _poll_status(client, base, timeout_s=args.timeout)
    return 0


def run_models_status(args: argparse.Namespace) -> int:
    base = args.base_url or DEFAULT_API_BASE
    with httpx.Client() as client:
        resp = client.get(_join_url(base, "/api/models/profile"), timeout=10)
        if resp.status_code >= 400:
            print(f"Failed to fetch status: HTTP {resp.status_code}")
            return 1
        data = resp.json()
        _print_status(data.get("status") or {})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LocalPro CLI")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE, help="API base URL")
    subparsers = parser.add_subparsers(dest="command")

    models = subparsers.add_parser("models", help="Model management")
    models_sub = models.add_subparsers(dest="models_cmd")

    profile = models_sub.add_parser("profile", help="Profile discovered models")
    profile.add_argument("--force", action="store_true", help="Re-profile even if cached")
    profile.add_argument("--wait", action="store_true", help="Wait for profiling to finish")
    profile.add_argument("--timeout", type=int, default=900, help="Max wait seconds")
    profile.add_argument("models", nargs="*", help="Limit to specific model keys")

    status = models_sub.add_parser("status", help="Show profiling status")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models" and args.models_cmd == "profile":
        return run_models_profile(args)
    if args.command == "models" and args.models_cmd == "status":
        return run_models_status(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
