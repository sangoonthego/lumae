"""Compact local client for the persistent visual review session."""
from __future__ import annotations

import argparse
import json
from urllib.parse import unquote

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["next", "events", "verify", "stats"])
    parser.add_argument("encoded_payload", nargs="?", default="%7B%7D")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    payload = json.loads(unquote(args.encoded_payload))
    result = requests.post(f"http://127.0.0.1:{args.port}/{args.action}", json=payload, timeout=900)
    print(json.dumps(result.json(), ensure_ascii=False))
    result.raise_for_status()


if __name__ == "__main__":
    main()
