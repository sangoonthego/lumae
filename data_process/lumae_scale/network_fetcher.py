"""Restricted localhost bridge for approved new AdsQA source downloads.

The network-enabled process needs no access to protected D200 manifests. The
annotation process retains all eligibility and content checks.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlparse

from .models import ROOT
from .source_pool import download

SLOTS = threading.BoundedSemaphore(6)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            if self.path != "/fetch":
                raise ValueError("Unknown endpoint")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            vid, url = body["source_video_id"], body["source_url"]
            parsed = urlparse(url)
            if not re.fullmatch(r"[a-f0-9]{32}", vid) or parsed.scheme != "https" or parsed.hostname != "video.adsoftheworld.com":
                raise ValueError("Only valid AdsQA source IDs and HTTPS URLs are allowed")
            path = ROOT / "local_data/raw/adsqa/videos" / (vid + ".mp4")
            with SLOTS:
                result = download({"target_name": vid + ".mp4", "source_url": url}, video_dir=path.parent)
            payload, status = {"status": "READY", "path": str(result), "bytes": result.stat().st_size}, 200
        except Exception as exc:
            import requests
            status = 404 if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code in (404,410) else 503
            payload = {"status": "ERROR", "kind": type(exc).__name__, "reason": str(exc)}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serve", choices=["serve"])
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__":
    main()
