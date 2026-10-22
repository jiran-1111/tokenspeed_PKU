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

"""Run only the original 16 isolated Stage B engine trajectories on one device."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.course_project.core import build_serve_command, load_model_config
from scripts import collect_homework1_references as collector
from scripts.collect_homework1_validation_reference import validation_cases
from scripts.run_session_kv import _engine_cli_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--parent-source-manifest", type=Path, required=True)
    args = parser.parse_args()
    if (
        collector.sha_file(args.parent_source_manifest)
        != "f44d5c85b1d9bfe30d3e04177fa6d4b5daf773564ee0a5611c07179fa2acb34e"
    ):
        raise ValueError("Frozen parent source manifest mismatch")
    output = collector.new_output_path(args.output)
    if (
        collector.sha_file(args.manifest)
        != "23dbfc6e21d99b674f0aa40a95374c4a215a7ec838f715692eec12d07f3f22fb"
    ):
        raise ValueError("Frozen manifest file mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = load_model_config(args.config)
    if (
        config["model_revision"] != "2fc06364715b967f1860aea9cf38778875588b17"
        or config["tp_size"] != 1
        or config["dtype"] != "bfloat16"
    ):
        raise ValueError("Frozen model configuration mismatch")
    cases = validation_cases(
        manifest,
        revision=config["model_revision"],
        expected_sha="f78c051acc44a969210824200d0b3de17dbc32eaca6643b8124383c47f1d4936",
        vocab_size=248320,
    )
    identity = json.loads((args.reference / "identity.json").read_text())
    reference_result = json.loads((args.reference / "result.json").read_text())
    if (
        reference_result["case_count"] != 16
        or identity["weights"]["global_weight_sha256"]
        != "c2084362ae79d1d0f3d61a9781598296fae0944efe4e6fd04888a48692c57884"
    ):
        raise ValueError("Primary reference incomplete or wrong projected weights")
    sources = json.loads(args.source_manifest.read_text())

    def check_sources() -> None:
        for name, sha in sources.items():
            if collector.sha_file(REPO_ROOT / name) != sha:
                raise ValueError(f"Source snapshot mismatch: {name}")
        for name, sha in identity["source_sha256"].items():
            path = Path(name)
            if (
                path.is_relative_to((REPO_ROOT / config["local_model_path"]).resolve())
                and collector.sha_file(path) != sha
            ):
                raise ValueError("Checkpoint identity changed")

    check_sources()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.logical_device):
        raise ValueError(
            "Requires the explicitly selected, freshly checked logical device exclusively"
        )
    output.mkdir(parents=True, exist_ok=False)
    arm_path = output / "arm.json"
    os.environ["TOKENSPEED_STAGE_B_ARM"] = str(arm_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "scripts/diagnostics/head_capture"),
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    # Eager is explicit: each real forward must execute the capture and guard.
    config = {
        **config,
        "enforce_eager": True,
        "overlap_schedule": False,
        "prefix_caching": False,
        "disable_kvstore": True,
        "prometheus_port": 58481,
        "tokenspeed_metrics_port": 58482,
    }
    collector.write_json(
        output / "protocol.json",
        {
            "config": config,
            "manifest_file_sha256": collector.sha_file(args.manifest),
            "entrypoint_sha256": collector.sha_file(Path(__file__)),
            "hook_sha256": collector.sha_file(
                REPO_ROOT / "scripts/homework1_stage_b_capture.py"
            ),
            "source_manifest_sha256": collector.sha_file(args.source_manifest),
            "reference_identity_sha256": collector.sha_file(
                args.reference / "identity.json"
            ),
            "diagnostic_only": True,
            "logical_device": args.logical_device,
            "parent_source_manifest_sha256": collector.sha_file(
                args.parent_source_manifest
            ),
            "continuation_steps": 16,
            "greedy_semantics": "unmodified logits argmax at each forced input frontier; not free-running greedy",
        },
    )
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    cli = _engine_cli_args(build_serve_command(config, host="127.0.0.1", port=28480))
    try:
        with Engine(server_args=prepare_server_args(cli)) as engine:
            for index, case in enumerate(cases):
                case_output = output / f"case_{index:02d}"
                case_output.mkdir(exist_ok=False)
                collector.write_json(arm_path, {**case, "output": str(case_output)})
                try:
                    result = engine.generate(
                        input_ids=case["input_ids"],
                        sampling_params={
                            "temperature": 0,
                            "max_new_tokens": 17,
                            "ignore_eos": True,
                        },
                    )
                finally:
                    arm_path.unlink(missing_ok=True)
                collector.write_json(case_output / "response.json", result)
                if (
                    result["output_ids"][:16] != case["continuation_ids"]
                    or len(result["output_ids"]) != 17
                ):
                    raise ValueError(
                        "Engine did not consume the original fixed continuation"
                    )
                if (
                    len(list(case_output.glob("hidden_*.pt"))) != 17
                    or len(list(case_output.glob("step_*.pt"))) != 17
                    or len(list(case_output.glob("step_*.json"))) != 17
                ):
                    raise ValueError("Incomplete capture or sampling records")
                print(f"Collected {index + 1}/16: {case['case_id']}", flush=True)
        check_sources()
        collector.write_json(
            output / "result.json",
            {"status": "collected", "case_count": 16, "vector_count": 272},
        )
    except Exception as error:
        collector.write_json(
            output / "failed.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
