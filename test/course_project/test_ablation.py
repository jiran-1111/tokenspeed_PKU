import inspect
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.course_project import ablation
from benchmarks.course_project.ablation import build_ablation_configs
from scripts.run_ablation import _wait_for_server


class TestAblationConfigs(unittest.TestCase):
    def test_mixed_batch_stays_disabled_for_ascend_variants(self) -> None:
        base = {
            "chunked_prefill_size": 4096,
            "prefix_caching": True,
            "mixed_batch": False,
            "overlap_schedule": True,
        }

        variants = build_ablation_configs(base)

        self.assertTrue(
            all(not variant["config"]["mixed_batch"] for variant in variants)
        )

    def test_kvstore_is_fixed_off_for_prefix_cache_ablation(self) -> None:
        base = {
            "chunked_prefill_size": 4096,
            "prefix_caching": True,
            "mixed_batch": False,
            "overlap_schedule": True,
        }

        variants = build_ablation_configs(base)

        self.assertTrue(
            all(variant["config"].get("disable_kvstore") for variant in variants)
        )

    def test_chunked_prefill_variant_uses_requested_size(self) -> None:
        self.assertIn(
            "chunked_prefill_size",
            inspect.signature(ablation.build_ablation_configs).parameters,
        )
        base = {
            "chunked_prefill_size": 4096,
            "max_model_len": 4096,
            "prefix_caching": True,
            "mixed_batch": False,
            "overlap_schedule": True,
        }

        variants = build_ablation_configs(base, chunked_prefill_size=512)

        self.assertEqual(
            [variant["config"]["chunked_prefill_size"] for variant in variants],
            [4096, 4096, 512, 512],
        )

    def test_waits_for_gateway_readiness(self) -> None:
        requested_paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                requested_paths.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format: str, *_args: object) -> None:
                pass

        class RunningProcess:
            def poll(self) -> None:
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            _wait_for_server(
                RunningProcess(),
                "127.0.0.1",
                server.server_port,
                timeout_seconds=1,
            )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

        self.assertEqual(requested_paths, ["/readiness"])

    def test_variant_ports_keep_gateway_and_engine_metrics_distinct(self) -> None:
        config = {}
        base_config = {
            "prometheus_port": 39141,
            "tokenspeed_metrics_port": 39142,
        }

        self.assertTrue(hasattr(ablation, "configure_ablation_ports"))
        ablation.configure_ablation_ports(config, base_config, variant_index=1)

        self.assertEqual(config["prometheus_port"], 39142)
        self.assertEqual(config["tokenspeed_metrics_port"], 39143)
        self.assertNotEqual(
            config["prometheus_port"], config["tokenspeed_metrics_port"]
        )

    def test_each_upstream_toggle_changes_one_optimization_stage(self) -> None:
        base = {
            "chunked_prefill_size": 4096,
            "prefix_caching": True,
            "mixed_batch": True,
            "overlap_schedule": True,
            "enforce_eager": False,
        }

        variants = build_ablation_configs(base)

        self.assertEqual(
            [variant["name"] for variant in variants], ["E0", "E1", "E2", "E3"]
        )
        self.assertEqual(variants[0]["config"]["chunked_prefill_size"], 4096)
        self.assertFalse(variants[0]["config"]["prefix_caching"])
        self.assertFalse(variants[0]["config"]["mixed_batch"])
        self.assertFalse(variants[0]["config"]["overlap_schedule"])
        self.assertFalse(variants[1]["config"]["mixed_batch"])
        self.assertTrue(variants[1]["config"]["overlap_schedule"])
        self.assertEqual(variants[1]["config"]["chunked_prefill_size"], 4096)
        self.assertEqual(variants[2]["config"]["chunked_prefill_size"], 4096)
        self.assertFalse(variants[2]["config"]["prefix_caching"])
        self.assertTrue(variants[3]["config"]["prefix_caching"])

    def test_ablation_script_help_is_available_without_npu_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "run_ablation.py"), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--runs", result.stdout)
        self.assertIn("--chunked-prefill-size", result.stdout)


if __name__ == "__main__":
    unittest.main()
