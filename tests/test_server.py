"""Exercise the real HTTP boundary without loading model weights."""

import http.client
import json
import threading
import unittest

from deciphish.server import MAX_REQUEST_BYTES, create_server


class FakeDetector:
    device = "cpu"
    profile = "phreshphish"

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, url, html):
        self.calls.append((url, html))
        if html == "wait":
            self.entered.set()
            self.release.wait(5)
        return {"phishing_score": 0.9 if "phish" in html else 0.1}


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector()
        self.server = create_server(self.detector, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method="POST", path="/check", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=8)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            connection.close()

    def snapshot(self, html="phish"):
        return self.request(body=json.dumps({"url": "https://example.org/", "html": html}),
                            headers={"Content-Type": "application/json"})

    def test_health_and_model_contract(self):
        status, headers, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["model_ready"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.snapshot()[2]["label"], "phishing")
        self.assertEqual(self.snapshot("ordinary")[2]["label"], "benign")
        self.assertEqual(self.detector.calls[0], ("https://example.org/", "phish"))

    def test_origin_and_host(self):
        for headers in ({"Origin": "https://evil.example"}, {"Host": "evil.example"},
                        {"Origin": "chrome-extension://[invalid"}):
            self.assertEqual(self.request("GET", "/health", headers=headers)[0], 403)
        status, headers, _ = self.request("OPTIONS", headers={"Origin": "chrome-extension://abc"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "chrome-extension://abc")
        self.assertEqual(self.detector.calls, [])

    def test_rejects_bad_input_without_inference(self):
        self.assertEqual(self.request(body="html")[0], 415)
        headers = {"Content-Type": "application/json"}
        for body in ("[]", "{", '{"url":"file:///test","html":"x"}',
                     '{"url":"https://example.org","html":""}'):
            self.assertEqual(self.request(body=body, headers=headers)[0], 400)
        self.assertEqual(self.request(body="", headers={**headers, "Content-Length": str(MAX_REQUEST_BYTES + 1)})[0], 413)
        self.assertEqual(self.detector.calls, [])

    def test_busy_model_is_not_reported_as_benign(self):
        result = []
        worker = threading.Thread(target=lambda: result.append(self.snapshot("wait")))
        worker.start()
        try:
            self.assertTrue(self.detector.entered.wait(3))
            status, _, body = self.snapshot()
            self.assertEqual((status, body), (503, {"error": "MODEL_BUSY"}))
        finally:
            self.detector.release.set()
            worker.join(5)
        self.assertEqual(result[0][0], 200)


if __name__ == "__main__":
    unittest.main()
