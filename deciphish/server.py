"""Loopback HTTP interface for browser URL/HTML snapshots.

The model is loaded once; prediction uses the same extractor, RACE encoder and
DCER readout as ``python -m deciphish predict``. No pages are fetched here.
"""

import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LOG = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class SnapshotDetector:
    def __init__(self, base_model, checkpoint, device="auto", profile="phreshphish"):
        import torch

        from .model import load_detector, load_tokenizer
        from .race import RaceEncoder

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.profile = profile
        self.encoder = RaceEncoder(load_tokenizer(base_model))
        self.model = load_detector(base_model, self.encoder.tokenizer, self.device, checkpoint)

    def __call__(self, url, html):
        from .evidence import make_view
        from .evaluation import predict

        view = make_view(url, html.encode("utf-8"), profile=self.profile)
        return predict(self.model, self.encoder, view, self.device)


def create_server(detector, port=8765):
    """Create a loopback-only service; accept an injectable detector for tests."""
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "DeciPhish"

        def log_message(self, fmt, *args):
            # Do not log page URLs or HTML from a user's browsing session.
            LOG.info(fmt, *args)

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def allowed(self):
            expected_port = self.server.server_port
            if self.headers.get("Host") not in {
                f"127.0.0.1:{expected_port}", f"localhost:{expected_port}"
            }:
                self.reply(403, {"error": "INVALID_HOST"})
                return False
            origin = self.headers.get("Origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    self.reply(403, {"error": "ORIGIN_NOT_ALLOWED"})
                    return False
                # Unpacked Chrome/Edge extension IDs vary between machines.
                if (parsed.scheme != "chrome-extension" or not parsed.hostname
                        or parsed.netloc != parsed.hostname or parsed.path
                        or parsed.query or parsed.fragment):
                    self.reply(403, {"error": "ORIGIN_NOT_ALLOWED"})
                    return False
            return True

        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            origin = self.headers.get("Origin", "")
            if status != 403 and origin.startswith("chrome-extension://"):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def do_OPTIONS(self):
            if self.allowed():
                self.reply(200 if self.path in {"/check", "/health"} else 404, {})

        def do_GET(self):
            if not self.allowed():
                return
            if self.path != "/health":
                self.reply(404, {"error": "NOT_FOUND"})
                return
            self.reply(200, {"model_ready": True, "device": str(detector.device),
                             "dtype": "bfloat16", "profile": detector.profile})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/check":
                self.reply(404, {"error": "NOT_FOUND"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "CONTENT_LENGTH_REQUIRED"})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "JSON_REQUIRED"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.reply(413 if length > MAX_REQUEST_BYTES else 400,
                           {"error": "INVALID_REQUEST_SIZE"})
                return
            try:
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected an object")
                url, html = data.get("url"), data.get("html")
                if not isinstance(url, str) or not isinstance(html, str) or not html.strip():
                    raise ValueError("URL and nonempty HTML strings are required")
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("Expected an HTTP(S) URL")
            except (ValueError, UnicodeError):
                self.reply(400, {"error": "INVALID_SNAPSHOT"})
                return
            # Bound the inference backlog. Callers can scan again when busy.
            if not model_lock.acquire(blocking=False):
                self.reply(503, {"error": "MODEL_BUSY"})
                return
            started = time.perf_counter()
            try:
                result = detector(url, html)
                score = float(result["phishing_score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid model score")
                self.reply(200, {"score": score,
                                 "label": "phishing" if score >= 0.5 else "benign",
                                 "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
            except Exception:
                LOG.exception("Snapshot inference failed")
                self.reply(500, {"error": "INFERENCE_FAILED"})
            finally:
                model_lock.release()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(base_model, checkpoint, *, port=8765, device="auto", profile="phreshphish"):
    logging.basicConfig(level=logging.INFO)
    detector = SnapshotDetector(base_model, checkpoint, device, profile)
    with create_server(detector, port) as server:
        LOG.info("Browser service ready at http://127.0.0.1:%d", server.server_port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
