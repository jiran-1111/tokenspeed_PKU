import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.course_project.profiler import (
    ProfilerControlClient,
    profiler_control_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _ControlHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append((self.path, body))
        payload = json.dumps({"success": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class TestProfilerControlClient(unittest.TestCase):
    def test_control_url_uses_sidecar_port(self) -> None:
        self.assertEqual(
            profiler_control_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8001",
        )
        self.assertEqual(
            profiler_control_url("https://example.invalid:8443/gateway/v1/"),
            "https://example.invalid:8444/gateway",
        )

    def setUp(self) -> None:
        _ControlHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ControlHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = ProfilerControlClient(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_start_sends_npu_profile_configuration(self) -> None:
        self.client.start(
            output_dir=Path("results/profiles/prefill"),
            profile_id="prefill-001",
        )

        path, body = _ControlHandler.requests[0]
        self.assertEqual(path, "/start_profile")
        self.assertEqual(body["activities"], ["CPU", "NPU", "MEM"])
        self.assertEqual(body["profile_id"], "prefill-001")
        self.assertTrue(body["record_shapes"])

    def test_stop_posts_to_stop_endpoint(self) -> None:
        self.client.stop()
        self.assertEqual(_ControlHandler.requests, [("/stop_profile", {})])

    def test_profiler_script_has_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_profiler.py"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prefill-heavy", result.stdout)


if __name__ == "__main__":
    unittest.main()
