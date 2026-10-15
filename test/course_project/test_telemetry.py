import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.course_project import telemetry
from benchmarks.course_project.telemetry import (
    add_metric_delta,
    parse_prometheus_metrics,
)


class TestTelemetry(unittest.TestCase):
    def test_aggregates_cache_and_scheduler_metrics_across_ranks(self) -> None:
        metrics = parse_prometheus_metrics(
            "\n".join(
                [
                    'tokenspeed:kv_cache_usage_perc{tp_rank="0"} 0.25',
                    'tokenspeed:kv_cache_usage_perc{tp_rank="1"} 0.5',
                    'tokenspeed:prefix_cache_hits_total{tp_rank="0"} 20',
                    'tokenspeed:prefix_cache_hits_total{tp_rank="1"} 30',
                    'tokenspeed:prompt_tokens_total{tp_rank="0"} 100',
                    'tokenspeed:prompt_tokens_total{tp_rank="1"} 150',
                    'tokenspeed:num_requests_running{tp_rank="0"} 2',
                    'tokenspeed:num_requests_running{tp_rank="1"} 3',
                ]
            )
        )

        self.assertEqual(metrics["kv_cache_usage_ratio"], 0.5)
        self.assertEqual(metrics["prefix_cache_hit_tokens"], 50.0)
        self.assertEqual(metrics["prompt_tokens"], 250.0)
        self.assertEqual(metrics["requests_running"], 5.0)

    def test_aggregates_inter_token_histogram_buckets_across_ranks(self) -> None:
        metrics = parse_prometheus_metrics(
            "\n".join(
                [
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.01",tp_rank="0"} 3',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.01",tp_rank="1"} 2',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.02",tp_rank="0"} 7',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.02",tp_rank="1"} 5',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="+Inf",tp_rank="0"} 9',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="+Inf",tp_rank="1"} 6',
                    'tokenspeed:inter_token_latency_seconds_count{tp_rank="0"} 9',
                    'tokenspeed:inter_token_latency_seconds_count{tp_rank="1"} 6',
                    'tokenspeed:inter_token_latency_seconds_sum{tp_rank="0"} 0.12',
                    'tokenspeed:inter_token_latency_seconds_sum{tp_rank="1"} 0.08',
                ]
            )
        )

        self.assertEqual(
            metrics.get("itl_histogram_buckets"),
            {0.01: 5.0, 0.02: 12.0, float("inf"): 15.0},
        )
        self.assertEqual(metrics.get("itl_histogram_count"), 15.0)
        self.assertAlmostEqual(metrics.get("itl_histogram_sum_seconds", 0), 0.2)

    def test_summarizes_itl_from_histogram_deltas_and_bounds_infinite_bucket(
        self,
    ) -> None:
        before = {
            "itl_histogram_buckets": {
                0.01: 10.0,
                0.02: 20.0,
                0.04: 40.0,
                0.08: 50.0,
                float("inf"): 50.0,
            },
            "itl_histogram_count": 50.0,
            "itl_histogram_sum_seconds": 1.5,
        }
        after = {
            "itl_histogram_buckets": {
                0.01: 10.0,
                0.02: 30.0,
                0.04: 75.0,
                0.08: 90.0,
                float("inf"): 110.0,
            },
            "itl_histogram_count": 110.0,
            "itl_histogram_sum_seconds": 3.9,
        }

        summary = add_metric_delta({}, before, after)

        self.assertEqual(summary.get("itl_observation_count"), 60.0)
        self.assertEqual(summary.get("itl_mean_ms"), 40.0)
        self.assertAlmostEqual(summary.get("itl_p50_ms"), 36.0)
        # The p95 lies in +Inf, so the estimate is bounded by the highest finite bucket.
        self.assertEqual(summary.get("itl_p95_ms"), 80.0)
        self.assertEqual(
            summary.get("itl_observation"),
            "server_prometheus_histogram_delta_inf_bucket_lower_bound",
        )

    def test_itl_histogram_delta_handles_a_reset_on_one_rank(self) -> None:
        def metrics(
            rank_samples: dict[str, tuple[float, float, tuple[float, float, float]]],
        ) -> dict:
            lines = []
            for rank, (count, total, buckets) in rank_samples.items():
                for bound, value in zip(("0.01", "0.02", "+Inf"), buckets):
                    lines.append(
                        "tokenspeed:inter_token_latency_seconds_bucket"
                        f'{{le="{bound}",tp_rank="{rank}"}} {value}'
                    )
                lines.extend(
                    [
                        f'tokenspeed:inter_token_latency_seconds_count{{tp_rank="{rank}"}} {count}',
                        f'tokenspeed:inter_token_latency_seconds_sum{{tp_rank="{rank}"}} {total}',
                    ]
                )
            return parse_prometheus_metrics("\n".join(lines))

        before = metrics(
            {
                "0": (100, 5.0, (50, 100, 100)),
                "1": (100, 5.0, (50, 100, 100)),
            }
        )
        after = metrics(
            {
                "0": (20, 1.0, (10, 20, 20)),
                "1": (110, 5.5, (55, 110, 110)),
            }
        )

        summary = add_metric_delta({}, before, after)

        self.assertEqual(summary.get("itl_observation_count"), 30.0)
        self.assertEqual(summary.get("itl_mean_ms"), 50.0)

    def test_itl_histogram_uses_zero_baseline_when_before_snapshot_is_missing(
        self,
    ) -> None:
        after = parse_prometheus_metrics(
            "\n".join(
                [
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.01",tp_rank="0"} 1',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="0.02",tp_rank="0"} 2',
                    'tokenspeed:inter_token_latency_seconds_bucket{le="+Inf",tp_rank="0"} 3',
                    'tokenspeed:inter_token_latency_seconds_count{tp_rank="0"} 3',
                    'tokenspeed:inter_token_latency_seconds_sum{tp_rank="0"} 0.06',
                ]
            )
        )

        summary = add_metric_delta({}, {}, after)

        self.assertEqual(summary.get("itl_observation_count"), 3.0)
        self.assertEqual(summary.get("itl_mean_ms"), 20.0)

    def test_no_new_itl_observations_remain_null_with_provenance(self) -> None:
        histogram = {
            "itl_histogram_buckets": {0.01: 0.0, 0.02: 0.0, float("inf"): 0.0},
            "itl_histogram_count": 0.0,
            "itl_histogram_sum_seconds": 0.0,
        }

        summary = add_metric_delta({}, histogram, histogram)

        self.assertIsNone(summary.get("itl_p50_ms"))
        self.assertIsNone(summary.get("itl_p95_ms"))
        self.assertEqual(summary.get("itl_observation_count"), 0.0)
        self.assertEqual(
            summary.get("itl_observation"),
            "server_prometheus_histogram_no_new_observations",
        )

    def test_missing_itl_histogram_remains_null_with_clear_provenance(self) -> None:
        summary = add_metric_delta({}, {}, {})

        self.assertIsNone(summary.get("itl_p50_ms"))
        self.assertIsNone(summary.get("itl_p95_ms"))
        self.assertEqual(
            summary.get("itl_observation"), "server_prometheus_histogram_unavailable"
        )

    def test_summary_records_cache_hit_delta_and_kv_usage(self) -> None:
        summary = add_metric_delta(
            {},
            {
                "kv_cache_usage_ratio": 0.1,
                "prefix_cache_hit_tokens": 10,
                "prompt_tokens": 100,
            },
            {
                "kv_cache_usage_ratio": 0.6,
                "prefix_cache_hit_tokens": 50,
                "prompt_tokens": 200,
            },
        )

        self.assertEqual(summary["kv_cache_usage_ratio"], 0.6)
        self.assertEqual(summary["prefix_cache_hit_tokens_delta"], 40)
        self.assertEqual(summary["prompt_tokens_delta"], 100)
        self.assertEqual(summary["prefix_cache_hit_ratio"], 0.4)

    def test_summary_records_peak_kv_usage_during_workload(self) -> None:
        summary = add_metric_delta(
            {},
            {"kv_cache_usage_ratio": 0.1},
            {"kv_cache_usage_ratio": 0.0},
            {"kv_cache_usage_ratio": 0.75},
        )

        self.assertEqual(summary["kv_cache_usage_ratio"], 0.0)
        self.assertEqual(summary["kv_cache_usage_ratio_peak"], 0.75)

    def test_server_metrics_sampler_returns_peak_observed_usage(self) -> None:
        sampler_type = getattr(telemetry, "ServerMetricsSampler", None)
        self.assertIsNotNone(sampler_type, "periodic metrics sampler is missing")
        observed_second_sample = threading.Event()
        samples = iter(
            [
                {"kv_cache_usage_ratio": 0.1, "requests_running": 1.0},
                {"kv_cache_usage_ratio": 0.75, "requests_running": 3.0},
            ]
        )

        def read_sample(*_args, **_kwargs):
            sample = next(samples)
            if sample["kv_cache_usage_ratio"] == 0.75:
                observed_second_sample.set()
            return sample

        with mock.patch.object(
            telemetry, "snapshot_server_metrics", side_effect=read_sample
        ):
            sampler = sampler_type("http://127.0.0.1:39142", interval_seconds=0.01)
            sampler.start()
            self.assertTrue(observed_second_sample.wait(timeout=1))
            during = sampler.stop()

        self.assertEqual(during["kv_cache_usage_ratio"], 0.75)
        self.assertEqual(during["requests_running"], 3.0)

    def test_missing_npu_smi_is_a_supported_local_host(self) -> None:
        from unittest import mock

        from benchmarks.course_project.telemetry import capture_npu_smi

        with mock.patch(
            "benchmarks.course_project.telemetry.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertFalse(capture_npu_smi(Path("results/hardware/test.txt")))

    def test_npu_snapshot_queries_only_npu_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "npu.txt"
            completed = mock.Mock(returncode=0, stdout="NPU ID : 0\n")
            with mock.patch.object(
                telemetry.subprocess, "run", return_value=completed
            ) as run:
                self.assertTrue(telemetry.capture_npu_smi(output_path))

            self.assertEqual(
                run.call_args.args[0],
                ["npu-smi", "info", "-t", "usages", "-i", "0"],
            )
            self.assertEqual(output_path.read_text(encoding="utf-8"), completed.stdout)

    def test_exclusive_npu_snapshot_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "npu.txt"
            output_path.write_text("original snapshot", encoding="utf-8")
            with mock.patch.object(telemetry.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    telemetry.capture_npu_smi(output_path, exclusive=True)

            run.assert_not_called()
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "original snapshot"
            )


if __name__ == "__main__":
    unittest.main()
