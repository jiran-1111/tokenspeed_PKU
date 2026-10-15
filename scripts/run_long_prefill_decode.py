#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project import long_prefill_decode as workload
from benchmarks.course_project.core import (
    OpenAIChatClient,
    build_serve_command,
    load_model_config,
)
from benchmarks.course_project.telemetry import capture_npu_smi

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "raw" / "benchmark" / "long_prefill_decode"
)
MEASUREMENT_CONDITIONS = (
    "decode-only",
    "prefill-only-2048",
    "prefill-only-3584",
    "staged-2048",
    "staged-3584",
)
_CHUNK_ORDER_BY_ROUND = ((4096, 512), (512, 4096), (4096, 512))
_UTILIZATION_FIELDS = {
    "npu_utilization": "npu utilization",
    "aicore_utilization": "aicore usage rate",
    "aivector_utilization": "aivector usage rate",
    "hbm_bandwidth_utilization": "hbm bandwidth usage rate",
}
_IGNORED_USAGE_FIELDS = {
    "aicpu usage rate",
    "aicube usage rate",
    "ctrlcpu usage rate",
    "ddr usage rate",
    "ddr hugepages usage rate",
    "ddr bandwidth usage rate",
    "hbm usage rate",
}
_REDACTED_VALUE = "<REDACTED>"
_SENSITIVE_CONFIG_KEY_PARTS = {
    "auth",
    "authorization",
    "oauth",
    "password",
    "passwd",
    "passphrase",
    "secret",
    "credential",
    "credentials",
    "bearer",
    "key",
}
_SENSITIVE_COMPACT_KEY_MARKERS = (
    "authorization",
    "password",
    "passwd",
    "passphrase",
    "secret",
    "credential",
    "privatekey",
)
_COMPACT_CREDENTIAL_PREFIXES = (
    "api",
    "hf",
    "huggingface",
    "access",
    "refresh",
    "bearer",
)
_NON_SECRET_TOKEN_CONFIG_KEYS = {
    "maxtokens",
    "maxnewtokens",
    "outputtokens",
    "prompttokens",
    "numtokens",
    "maxnumbatchedtokens",
    "tokenizer",
}


def build_launch_schedule() -> list[dict]:
    """Return the fixed three-pair service launch order."""
    return [
        {"round": round_number, "order": order, "chunk_size": chunk_size}
        for round_number, chunk_order in enumerate(_CHUNK_ORDER_BY_ROUND, start=1)
        for order, chunk_size in enumerate(chunk_order, start=1)
    ]


def _resolve_inside_project(path: Path, label: str, *, strict: bool = False) -> Path:
    resolved = path.resolve(strict=strict)
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside the project tree") from error
    return resolved


def _resolve_model_path(config: dict, *, require_existing: bool) -> Path:
    configured_path = config.get("local_model_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("long prefill benchmark requires local_model_path")
    path = Path(configured_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = _resolve_inside_project(
        path, "local_model_path", strict=require_existing
    )
    if require_existing and not resolved.is_dir():
        raise ValueError("local_model_path must be an existing directory")
    return resolved


def _is_sensitive_config_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    compact_key = re.sub(r"[^a-z0-9]+", "", key.lower())
    if any(marker in compact_key for marker in _SENSITIVE_COMPACT_KEY_MARKERS):
        return True
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    separated = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", separated)
    parts = re.findall(r"[a-z0-9]+", separated.lower())
    if _SENSITIVE_CONFIG_KEY_PARTS.intersection(parts):
        return True
    if compact_key in _NON_SECRET_TOKEN_CONFIG_KEYS:
        return False
    # Treat standalone or compact-edge "token(s)" labels as credentials.
    token_candidate = compact_key.replace("tokenizer", "")
    if (
        {"token", "tokens"}.intersection(parts)
        or token_candidate.startswith(("token", "tokens"))
        or token_candidate.endswith(("token", "tokens"))
    ):
        return True
    prefixes = "|".join(_COMPACT_CREDENTIAL_PREFIXES)
    if re.search(
        rf"^(?:{prefixes})[a-z0-9]*(?:key|secret|credential|password)",
        compact_key,
    ):
        return True
    if "authheader" in compact_key:
        return True
    adjacent_pairs = set(zip(parts, parts[1:]))
    return bool(
        adjacent_pairs.intersection(
            {
                ("api", "key"),
                ("api", "token"),
                ("access", "token"),
                ("refresh", "token"),
                ("bearer", "token"),
                ("private", "key"),
            }
        )
    )


def _sanitize_persisted_config(value):
    if isinstance(value, dict):
        return {
            key: (
                _REDACTED_VALUE
                if _is_sensitive_config_key(key)
                else _sanitize_persisted_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_persisted_config(item) for item in value]
    return value


def load_local_tokenizer(config: dict):
    return workload.load_local_tokenizer(config)


def parse_npu0_utilization(snapshot: str) -> dict[str, float]:
    """Parse only the reported NPU 0 usage fields; reject unknown usage data."""
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise ValueError("NPU0 usage snapshot is empty or unreadable")

    npu_ids = []
    values: dict[str, list[float]] = {key: [] for key in _UTILIZATION_FIELDS}
    names = {name: key for key, name in _UTILIZATION_FIELDS.items()}

    for line in snapshot.splitlines():
        if ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        normalized = re.sub(r"[^a-z0-9]+", " ", label.strip().lower()).strip()
        value_text = raw_value.strip()
        if normalized == "npu id":
            if not re.fullmatch(r"\d+", value_text):
                raise ValueError("NPU0 usage snapshot has an invalid NPU ID")
            npu_ids.append(int(value_text))
            continue

        metric_key = names.get(normalized)
        if metric_key is None:
            if normalized in _IGNORED_USAGE_FIELDS:
                continue
            if "utilization" in normalized or "usage rate" in normalized:
                raise ValueError(f"unknown NPU0 utilization field: {label.strip()}")
            continue

        match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?", value_text)
        if match is None:
            raise ValueError(f"unparseable NPU0 utilization field: {label.strip()}")
        value = float(match.group(1))
        if not 0 <= value <= 100:
            raise ValueError(f"invalid NPU0 utilization value: {label.strip()}")
        values[metric_key].append(value)

    if not npu_ids or any(npu_id != 0 for npu_id in npu_ids):
        raise ValueError("NPU usage snapshot does not identify NPU 0 only")
    missing = [key for key, readings in values.items() if not readings]
    if missing:
        raise ValueError(
            "NPU0 usage snapshot is missing utilization fields: " + ", ".join(missing)
        )
    return {key: max(readings) for key, readings in values.items()}


def require_npu0_idle(snapshot: str) -> dict[str, float]:
    metrics = parse_npu0_utilization(snapshot)
    busy = {key: value for key, value in metrics.items() if value != 0}
    if busy:
        fields = ", ".join(f"{key}={value:g}%" for key, value in busy.items())
        raise ValueError(f"NPU0 is busy: {fields}")
    return metrics


def check_port_available(host: str, port: int) -> bool:
    """Check the configured bind address without discovering or killing PIDs."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        if sys.platform != "win32":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError:
            return False
    return True


def _check_listener_ports_available(config: dict, host: str, port: int) -> None:
    listeners = [
        ("API service", host, port),
        ("Prometheus exporter", "0.0.0.0", config["prometheus_port"]),
    ]
    if config["enable_metrics"]:
        listeners.append(
            ("TokenSpeed metrics", "127.0.0.1", config["tokenspeed_metrics_port"])
        )

    ports: dict[int, str] = {}
    for name, _bind_host, bind_port in listeners:
        if bind_port in ports:
            raise OSError(f"{ports[bind_port]} and {name} both use port {bind_port}")
        ports[bind_port] = name

    for name, bind_host, bind_port in listeners:
        if not check_port_available(bind_host, bind_port):
            raise OSError(f"{name} port {bind_port} on {bind_host} is not available")


def client_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1" if host == "0.0.0.0" else "::1"
    return host


def _url_authority(host: str, port: int) -> str:
    selected_host = client_host(host)
    if ":" in selected_host:
        return f"[{selected_host}]:{port}"
    return f"{selected_host}:{port}"


def service_base_url(host: str, port: int) -> str:
    return f"http://{_url_authority(host, port)}/v1"


def readiness_url(host: str, port: int) -> str:
    return f"http://{_url_authority(host, port)}/readiness"


def wait_for_readiness(
    process: subprocess.Popen,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = readiness_url(host, port)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"TokenSpeed exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"TokenSpeed did not become ready at {url}")


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex}"


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _write_jsonl(output, value: dict) -> None:
    output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    output.flush()


def _stop_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _prompt(tokenizer, target_tokens: int, seed: str) -> list[dict[str, str]]:
    return workload.build_exact_messages(tokenizer, target_tokens, seed)


def _trial_spec(
    *,
    run_id: str,
    condition: str,
    prefill_tokens: int,
    round_number: int,
    chunk_size: int,
    execution_order: int,
) -> workload.TrialSpec:
    return workload.TrialSpec(
        trial_id=f"{condition}-r{round_number}-c{chunk_size}",
        condition=condition,
        prefill_tokens=prefill_tokens,
        repetition=round_number,
        chunk_size=chunk_size,
        execution_order=execution_order,
    )


def _record_trial(
    *,
    trial: dict,
    trial_spec: workload.TrialSpec,
    run_id: str,
    pair_id: str | None,
    round_number: int,
    config: dict,
    config_sha256: str,
    prompt_hashes: dict[str, str],
    requests_output,
    trials_output,
) -> None:
    provenance = {
        "run_id": run_id,
        "pair_id": pair_id,
        "round": round_number,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "served_model_name": config["served_model_name"],
        "config_sha256": config_sha256,
        "prompt_recipe_version": workload.PROMPT_RECIPE_VERSION,
    }
    for request in trial["requests"]:
        request_role = request["request_role"]
        request_record = {
            **request,
            **provenance,
            "request_id": uuid4().hex,
            "trial_id": trial_spec.trial_id,
            "condition": trial_spec.condition,
            "repetition": trial_spec.repetition,
            "chunk_size": trial_spec.chunk_size,
            "prompt_sha256": prompt_hashes[request_role],
        }
        _write_jsonl(requests_output, request_record)
    _write_jsonl(
        trials_output,
        {
            **trial,
            **provenance,
            "pair_id": pair_id,
            "prompt_hashes": prompt_hashes,
        },
    )


def _run_one_trial(
    *,
    client,
    tokenizer,
    config: dict,
    run_id: str,
    launch: dict,
    condition: str,
    execution_order: int,
    requests_output,
    trials_output,
) -> dict:
    round_number = launch["round"]
    chunk_size = launch["chunk_size"]
    model = config["served_model_name"]
    seed_prefix = f"{run_id}:round-{round_number}"
    decode_messages = _prompt(tokenizer, 256, f"{seed_prefix}:decode")
    pair_id = f"{condition}-r{round_number}"

    if condition == "decode-only":
        prefill_tokens = 0
        prefill_messages = None
    else:
        prefill_tokens = int(condition.rsplit("-", 1)[1])
        prefill_messages = _prompt(
            tokenizer,
            prefill_tokens,
            f"{seed_prefix}:{condition}",
        )

    spec = _trial_spec(
        run_id=run_id,
        condition=condition,
        prefill_tokens=prefill_tokens,
        round_number=round_number,
        chunk_size=chunk_size,
        execution_order=execution_order,
    )
    if condition == "decode-only":
        trial = workload.run_single_trial(
            client=client,
            trial_spec=spec,
            request_role="decode",
            model=model,
            messages=decode_messages,
            expected_input_tokens=256,
            expected_output_tokens=128,
        )
        prompt_hashes = {"decode": workload.prompt_sha256(decode_messages)}
    elif condition.startswith("prefill-only-"):
        trial = workload.run_single_trial(
            client=client,
            trial_spec=spec,
            request_role="prefill",
            model=model,
            messages=prefill_messages,
            expected_input_tokens=prefill_tokens,
            expected_output_tokens=1,
        )
        prompt_hashes = {"prefill": workload.prompt_sha256(prefill_messages)}
    else:
        trial = workload.run_staged_trial(
            client=client,
            trial_spec=spec,
            model=model,
            decode_messages=decode_messages,
            prefill_messages=prefill_messages,
            expected_decode_input_tokens=256,
            expected_decode_output_tokens=128,
            expected_prefill_input_tokens=prefill_tokens,
            expected_prefill_output_tokens=1,
        )
        prompt_hashes = {
            "decode": workload.prompt_sha256(decode_messages),
            "prefill": workload.prompt_sha256(prefill_messages),
        }

    _record_trial(
        trial=trial,
        trial_spec=spec,
        run_id=run_id,
        pair_id=pair_id,
        round_number=round_number,
        config=config,
        config_sha256=hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        prompt_hashes=prompt_hashes,
        requests_output=requests_output,
        trials_output=trials_output,
    )
    return trial


def _run_warmup(
    *,
    client,
    tokenizer,
    config: dict,
    run_id: str,
    launch: dict,
    condition: str,
    target_tokens: int,
    requests_output,
    trials_output,
) -> dict:
    messages = _prompt(
        tokenizer,
        target_tokens,
        f"{run_id}:round-{launch['round']}:warmup:{condition}",
    )
    is_decode = condition == "warmup-decode"
    spec = _trial_spec(
        run_id=run_id,
        condition=condition,
        prefill_tokens=0 if is_decode else target_tokens,
        round_number=launch["round"],
        chunk_size=launch["chunk_size"],
        execution_order=0 if is_decode else 1,
    )
    trial = workload.run_single_trial(
        client=client,
        trial_spec=spec,
        request_role="decode" if is_decode else "prefill",
        model=config["served_model_name"],
        messages=messages,
        expected_input_tokens=target_tokens,
        expected_output_tokens=128 if is_decode else 1,
    )
    role = "decode" if is_decode else "prefill"
    _record_trial(
        trial=trial,
        trial_spec=spec,
        run_id=run_id,
        pair_id=None,
        round_number=launch["round"],
        config=config,
        config_sha256=hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        prompt_hashes={role: workload.prompt_sha256(messages)},
        requests_output=requests_output,
        trials_output=trials_output,
    )
    if trial["status"] != "valid":
        raise RuntimeError(f"{condition} warmup failed: {trial['invalid_reason']}")
    return trial


def _print_schedule(config: dict) -> None:
    print(
        "Dry run: no model load, tokenizer, output files, NPU, server, or HTTP requests."
    )
    for launch in build_launch_schedule():
        variant_config = workload.build_config_variants(config)[launch["chunk_size"]]
        print(
            f"round={launch['round']} order={launch['order']} "
            f"chunk_size={launch['chunk_size']} "
            f"prefix_caching={str(variant_config['prefix_caching']).lower()} "
            f"conditions={','.join(MEASUREMENT_CONDITIONS)}"
        )


def run(args: argparse.Namespace) -> Path | None:
    config_path = _resolve_inside_project(Path(args.config), "config path", strict=True)
    if not config_path.is_file():
        raise ValueError("config path must be an existing file")
    config = load_model_config(config_path)
    variants = workload.build_config_variants(config)
    _resolve_model_path(config, require_existing=not args.dry_run)
    output_root = _resolve_inside_project(Path(args.output_root), "output root")

    if args.dry_run:
        _print_schedule(config)
        return None
    if not args.confirm_npu0_idle:
        raise ValueError("actual NPU run requires --confirm-npu0-idle")
    if args.startup_timeout <= 0:
        raise ValueError("startup-timeout must be greater than zero")

    tokenizer = load_local_tokenizer(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = _resolve_inside_project(output_root, "output root", strict=True)
    run_id = new_run_id()
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"run directory already exists: {run_dir}") from None

    config_file_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[1]
    _write_json_exclusive(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "config_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_file_sha256": config_file_sha256,
            "benchmark_source_sha256": {
                relative: hashlib.sha256(
                    (source_root / relative).read_bytes()
                ).hexdigest()
                for relative in (
                    "scripts/run_long_prefill_decode.py",
                    "benchmarks/course_project/long_prefill_decode.py",
                    "benchmarks/course_project/core.py",
                    "benchmarks/course_project/runner.py",
                    "benchmarks/course_project/telemetry.py",
                )
            },
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "host": args.host,
            "port": args.port,
            "schedule": build_launch_schedule(),
            "conditions": list(MEASUREMENT_CONDITIONS),
        },
    )

    with ExitStack() as stack:
        requests_output = stack.enter_context(
            (run_dir / "requests.jsonl").open("x", encoding="utf-8")
        )
        trials_output = stack.enter_context(
            (run_dir / "trials.jsonl").open("x", encoding="utf-8")
        )
        for launch in build_launch_schedule():
            variant = variants[launch["chunk_size"]]
            config_sha256 = hashlib.sha256(
                json.dumps(variant, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            config_snapshot = {
                "schema_version": 1,
                "run_id": run_id,
                "round": launch["round"],
                "launch_order": launch["order"],
                "config_path": config_path.relative_to(REPO_ROOT).as_posix(),
                "config_file_sha256": config_file_sha256,
                "config_sha256": config_sha256,
                "config": _sanitize_persisted_config(variant),
            }
            _write_json_exclusive(
                run_dir
                / f"config-round-{launch['round']:02d}-chunk-{launch['chunk_size']}.json",
                config_snapshot,
            )

            hardware_path = (
                run_dir
                / f"npu0-before-round-{launch['round']:02d}-chunk-{launch['chunk_size']}.txt"
            )
            if not capture_npu_smi(hardware_path, exclusive=True):
                raise ValueError("could not read the NPU0 usage snapshot")
            require_npu0_idle(hardware_path.read_text(encoding="utf-8"))
            _check_listener_ports_available(variant, args.host, args.port)

            command = build_serve_command(variant, host=args.host, port=args.port)
            log_path = (
                run_dir
                / f"service-round-{launch['round']:02d}-chunk-{launch['chunk_size']}.log"
            )
            with log_path.open("x", encoding="utf-8") as log_file:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    log_file.write("startup_status=launch_failed\n")
                    raise
                try:
                    try:
                        wait_for_readiness(
                            process,
                            args.host,
                            args.port,
                            args.startup_timeout,
                        )
                    except Exception:
                        log_file.write("startup_status=failed\n")
                        raise
                    log_file.write("startup_status=ready\n")
                    log_file.flush()
                    client = OpenAIChatClient(service_base_url(args.host, args.port))
                    decode_warmup = _run_warmup(
                        client=client,
                        tokenizer=tokenizer,
                        config=variant,
                        run_id=run_id,
                        launch=launch,
                        condition="warmup-decode",
                        target_tokens=256,
                        requests_output=requests_output,
                        trials_output=trials_output,
                    )
                    if decode_warmup["requests"][0]["output_tokens"] != 128:
                        raise RuntimeError(
                            "Decode warmup must report exactly 128 output tokens"
                        )
                    _run_warmup(
                        client=client,
                        tokenizer=tokenizer,
                        config=variant,
                        run_id=run_id,
                        launch=launch,
                        condition="warmup-prefill-3584",
                        target_tokens=3584,
                        requests_output=requests_output,
                        trials_output=trials_output,
                    )
                    for condition_index, condition in enumerate(
                        MEASUREMENT_CONDITIONS, start=1
                    ):
                        trial = _run_one_trial(
                            client=client,
                            tokenizer=tokenizer,
                            config=variant,
                            run_id=run_id,
                            launch=launch,
                            condition=condition,
                            execution_order=condition_index,
                            requests_output=requests_output,
                            trials_output=trials_output,
                        )
                        if trial["status"] != "valid":
                            print(
                                f"Invalid trial retained: {trial['trial_id']} "
                                f"({trial['invalid_reason']})"
                            )
                finally:
                    try:
                        _stop_owned_process(process)
                    finally:
                        exit_status = process.poll()
                        log_file.write(
                            "exit_status="
                            f"{exit_status if exit_status is not None else 'unknown'}\n"
                        )

    print(run_dir)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure long-prefill interference with local Decode requests."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-npu0-idle", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
