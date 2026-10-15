from __future__ import annotations

import re
import subprocess
import threading
import urllib.error
import urllib.request
from math import inf, isfinite
from pathlib import Path
from typing import Any

_KV_BLOCK_METRICS = tuple(
    f"kv_lcm_blocks_{state}"
    for state in ("total", "active", "cached", "empty", "allocated")
)
_GAUGE_METRICS = (
    "kv_cache_usage_ratio",
    "requests_running",
    "requests_waiting",
) + _KV_BLOCK_METRICS
_METRIC_NAMES = {
    **{key: f"tokenspeed:{key}" for key in _KV_BLOCK_METRICS},
    "kv_cache_usage_ratio": "tokenspeed:kv_cache_usage_perc",
    "prefix_cache_hit_tokens": "tokenspeed:prefix_cache_hits_total",
    "prompt_tokens": "tokenspeed:prompt_tokens_total",
    "generation_tokens": "tokenspeed:generation_tokens_total",
    "requests_running": "tokenspeed:num_requests_running",
    "requests_waiting": "tokenspeed:num_requests_waiting",
}
_ITL_BUCKET_METRIC = "tokenspeed:inter_token_latency_seconds_bucket"
_ITL_COUNT_METRIC = "tokenspeed:inter_token_latency_seconds_count"
_ITL_SUM_METRIC = "tokenspeed:inter_token_latency_seconds_sum"
_SAMPLE_RE = re.compile(r"([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s|$)")
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


def parse_prometheus_metrics(payload: str) -> dict[str, Any]:
    samples: dict[str, list[float]] = {key: [] for key in _METRIC_NAMES}
    itl_buckets: dict[float, float] = {}
    itl_counts: list[float] = []
    itl_sums: list[float] = []
    itl_series: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        metric_name, raw_labels, raw_value = match.groups()
        value = float(raw_value)
        if metric_name == _ITL_BUCKET_METRIC:
            labels = dict(_LABEL_RE.findall(raw_labels or ""))
            if "le" in labels:
                upper_bound = float(labels["le"])
                itl_buckets[upper_bound] = itl_buckets.get(upper_bound, 0.0) + value
                series_id = tuple(
                    sorted((key, label) for key, label in labels.items() if key != "le")
                )
                series = itl_series.setdefault(series_id, {"buckets": {}})
                buckets = series["buckets"]
                buckets[upper_bound] = buckets.get(upper_bound, 0.0) + value
            continue
        if metric_name == _ITL_COUNT_METRIC:
            itl_counts.append(value)
            labels = dict(_LABEL_RE.findall(raw_labels or ""))
            series_id = tuple(sorted(labels.items()))
            series = itl_series.setdefault(series_id, {"buckets": {}})
            series["count"] = series.get("count", 0.0) + value
            continue
        if metric_name == _ITL_SUM_METRIC:
            itl_sums.append(value)
            labels = dict(_LABEL_RE.findall(raw_labels or ""))
            series_id = tuple(sorted(labels.items()))
            series = itl_series.setdefault(series_id, {"buckets": {}})
            series["sum_seconds"] = series.get("sum_seconds", 0.0) + value
            continue
        for key, expected_name in _METRIC_NAMES.items():
            if metric_name == expected_name:
                samples[key].append(value)

    result = {}
    for key, values in samples.items():
        if not values:
            continue
        if key == "kv_cache_usage_ratio":
            result[key] = max(values)
        else:
            result[key] = sum(values)
    if itl_buckets:
        result["itl_histogram_buckets"] = itl_buckets
    if itl_counts:
        result["itl_histogram_count"] = sum(itl_counts)
    if itl_sums:
        result["itl_histogram_sum_seconds"] = sum(itl_sums)
    if itl_series:
        result["itl_histogram_series"] = itl_series
    return result


def fetch_server_metrics(
    server_url: str, *, timeout_seconds: float = 10
) -> dict[str, Any]:
    metrics_url = server_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    request = urllib.request.Request(f"{metrics_url}/metrics")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return parse_prometheus_metrics(response.read().decode("utf-8"))


def snapshot_server_metrics(
    server_url: str, *, timeout_seconds: float = 10
) -> dict[str, Any]:
    try:
        return fetch_server_metrics(server_url, timeout_seconds=timeout_seconds)
    except (OSError, urllib.error.URLError):
        return {}


class ServerMetricsSampler:
    def __init__(self, server_url: str, *, interval_seconds: float = 0.25) -> None:
        self.server_url = server_url
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def stop(self) -> dict[str, float]:
        if self._started:
            self._stop_event.set()
            self._thread.join(timeout=2)
            self._started = False
        return self._peaks()

    def _peaks(self) -> dict[str, float]:
        peaks: dict[str, float] = {}
        for sample in self._samples:
            for key in _GAUGE_METRICS:
                if key in sample:
                    peaks[key] = max(peaks.get(key, sample[key]), sample[key])
        return peaks

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            sample = snapshot_server_metrics(self.server_url, timeout_seconds=1)
            if sample:
                self._samples.append(sample)
            if self._stop_event.wait(self.interval_seconds):
                break


def add_metric_delta(
    summary: dict,
    before: dict,
    after: dict,
    during: dict | None = None,
) -> dict:
    before = before or {}
    after = after or {}
    during = during or {}
    summary["server_metrics_available"] = bool(before or after or during)
    for key in _GAUGE_METRICS:
        if key in after:
            summary[key] = after[key]
        observed = [sample[key] for sample in (before, during, after) if key in sample]
        if observed:
            summary[f"{key}_peak"] = max(observed)
    for key in ("prefix_cache_hit_tokens", "prompt_tokens", "generation_tokens"):
        if key in before and key in after:
            summary[f"{key}_delta"] = max(0.0, after[key] - before[key])
    prompt_delta = summary.get("prompt_tokens_delta")
    hit_delta = summary.get("prefix_cache_hit_tokens_delta")
    if prompt_delta:
        summary["prefix_cache_hit_ratio"] = (hit_delta or 0.0) / prompt_delta
    _add_itl_histogram_delta(summary, before, after)
    return summary


def _counter_delta(before: float, after: float) -> float:
    return after - before if after >= before else max(0.0, after)


def _add_itl_histogram_delta(summary: dict, before: dict, after: dict) -> None:
    summary.update(
        {
            "itl_mean_ms": None,
            "itl_p50_ms": None,
            "itl_p95_ms": None,
            "itl_observation_count": None,
            "itl_observation": "server_prometheus_histogram_unavailable",
        }
    )
    before_series = before.get("itl_histogram_series")
    after_series = after.get("itl_histogram_series")
    if not after_series and not any(
        key in after
        for key in (
            "itl_histogram_buckets",
            "itl_histogram_count",
            "itl_histogram_sum_seconds",
        )
    ):
        return
    if isinstance(after_series, dict) and after_series:
        delta = _itl_series_delta(
            before_series if isinstance(before_series, dict) else {}, after_series
        )
    else:
        delta = _itl_aggregate_delta(before, after)
    if delta is None:
        summary["itl_observation"] = "server_prometheus_histogram_incomplete"
        return

    bucket_counts, count_delta, sum_delta = delta
    summary["itl_observation_count"] = count_delta
    if abs(count_delta - bucket_counts[inf]) > 1e-6:
        summary["itl_observation"] = "server_prometheus_histogram_incomplete"
        summary["itl_observation_count"] = None
        return
    if count_delta <= 0:
        summary["itl_observation"] = "server_prometheus_histogram_no_new_observations"
        summary["itl_observation_count"] = 0.0
        return

    summary["itl_mean_ms"] = sum_delta / count_delta * 1000
    p50_ms, p50_inf_bound = _histogram_quantile_ms(bucket_counts, count_delta, 0.50)
    p95_ms, p95_inf_bound = _histogram_quantile_ms(bucket_counts, count_delta, 0.95)
    summary["itl_p50_ms"] = p50_ms
    summary["itl_p95_ms"] = p95_ms
    summary["itl_observation"] = (
        "server_prometheus_histogram_delta_inf_bucket_lower_bound"
        if p50_inf_bound or p95_inf_bound
        else "server_prometheus_histogram_delta"
    )


def _itl_series_delta(
    before_series: dict,
    after_series: dict,
) -> tuple[dict[float, float], float, float] | None:
    if set(before_series) - set(after_series):
        return None

    aggregate_buckets: dict[float, float] = {}
    count_delta = 0.0
    sum_delta = 0.0
    common_bounds: set[float] | None = None
    for series_id, after in after_series.items():
        after_buckets = after.get("buckets")
        if (
            not isinstance(after_buckets, dict)
            or inf not in after_buckets
            or "count" not in after
            or "sum_seconds" not in after
        ):
            return None
        bounds = set(after_buckets)
        if common_bounds is None:
            common_bounds = bounds
        elif bounds != common_bounds:
            return None

        before = before_series.get(series_id)
        if before is None:
            before_buckets = {bound: 0.0 for bound in bounds}
            before_count = 0.0
            before_sum = 0.0
        else:
            before_buckets = before.get("buckets")
            if (
                not isinstance(before_buckets, dict)
                or set(before_buckets) != bounds
                or "count" not in before
                or "sum_seconds" not in before
            ):
                return None
            before_count = before["count"]
            before_sum = before["sum_seconds"]

        previous_count = 0.0
        for upper_bound in sorted(bounds):
            delta = _counter_delta(
                before_buckets[upper_bound], after_buckets[upper_bound]
            )
            cumulative_delta = max(previous_count, delta)
            aggregate_buckets[upper_bound] = (
                aggregate_buckets.get(upper_bound, 0.0) + cumulative_delta
            )
            previous_count = cumulative_delta
        count_delta += _counter_delta(before_count, after["count"])
        sum_delta += _counter_delta(before_sum, after["sum_seconds"])

    return aggregate_buckets, count_delta, sum_delta


def _itl_aggregate_delta(
    before: dict,
    after: dict,
) -> tuple[dict[float, float], float, float] | None:
    after_buckets = after.get("itl_histogram_buckets")
    if not isinstance(after_buckets, dict) or inf not in after_buckets:
        return None
    after_count = after.get("itl_histogram_count")
    after_sum = after.get("itl_histogram_sum_seconds")
    if after_count is None or after_sum is None:
        return None

    before_buckets = before.get("itl_histogram_buckets")
    if before_buckets is None:
        before_buckets = {bound: 0.0 for bound in after_buckets}
        before_count = 0.0
        before_sum = 0.0
    else:
        if not isinstance(before_buckets, dict) or set(before_buckets) != set(
            after_buckets
        ):
            return None
        before_count = before.get("itl_histogram_count")
        before_sum = before.get("itl_histogram_sum_seconds")
        if before_count is None or before_sum is None:
            return None

    bucket_counts: dict[float, float] = {}
    previous_count = 0.0
    for upper_bound in sorted(after_buckets):
        delta = _counter_delta(before_buckets[upper_bound], after_buckets[upper_bound])
        cumulative_delta = max(previous_count, delta)
        bucket_counts[upper_bound] = cumulative_delta
        previous_count = cumulative_delta
    return (
        bucket_counts,
        _counter_delta(before_count, after_count),
        _counter_delta(before_sum, after_sum),
    )


def _histogram_quantile_ms(
    buckets: dict[float, float], observation_count: float, quantile: float
) -> tuple[float | None, bool]:
    target = observation_count * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for upper_bound, cumulative_count in sorted(buckets.items()):
        if cumulative_count >= target:
            if not isfinite(upper_bound):
                return (
                    previous_bound * 1000 if previous_bound > 0 else None,
                    True,
                )
            bucket_count = cumulative_count - previous_count
            if bucket_count <= 0:
                return upper_bound * 1000, False
            fraction = (target - previous_count) / bucket_count
            estimate_seconds = previous_bound + fraction * (
                upper_bound - previous_bound
            )
            return estimate_seconds * 1000, False
        previous_bound = upper_bound
        previous_count = cumulative_count
    return None, False


def capture_npu_smi(output_path: Path, *, exclusive: bool = False) -> bool:
    """Save NPU 0 usage output when running on an Ascend host."""
    if exclusive and output_path.exists():
        raise FileExistsError(output_path)
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with output_path.open("x", encoding="utf-8") as output:
            output.write(result.stdout)
    else:
        output_path.write_text(result.stdout, encoding="utf-8")
    return True
