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

import importlib.util
import json
import re
from pathlib import Path

import pytest

from benchmarks.course_project.core import (
    build_serve_command,
    load_model_config,
    model_load_path,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ("0.8b", "0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("4b", "4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("9b", "9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("27b", "27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
)


def _config_path(model: str, tp_size: int) -> Path:
    path = ROOT / "configs/homework2" / f"qwen35_{model}_tp{tp_size}.yaml"
    assert path.is_file(), f"Missing HW2 configuration: {path.name}"
    return path


@pytest.mark.parametrize("model,size,revision", MODELS)
def test_tp_pair_preserves_pinned_identity_and_final_bytes(
    model: str, size: str, revision: str
) -> None:
    final_path = ROOT / "configs/final" / f"qwen35_{model}.yaml"
    final = load_model_config(final_path)
    pair = []
    for tp_size in (1, 2):
        path = _config_path(model, tp_size)
        config = load_model_config(path)
        assert config["model_id"] == f"Qwen/Qwen3.5-{size}"
        assert config["local_model_path"] == f"models/Qwen3.5-{size}"
        assert config["served_model_name"] == f"qwen35-{model}"
        assert config["model_revision"] == revision
        assert config == {**final, "tp_size": tp_size}
        expected, replacements = re.subn(
            rb'("tp_size": )[12]',
            lambda match: match[1] + str(tp_size).encode("ascii"),
            final_path.read_bytes(),
        )
        assert replacements == 1
        assert path.read_bytes() == expected
        pair.append(config)
    assert {key for key in pair[0] if pair[0][key] != pair[1][key]} == {"tp_size"}


@pytest.mark.parametrize("model,size,revision", MODELS)
@pytest.mark.parametrize("tp_size", (1, 2))
def test_serve_command_preserves_precision_and_four_request_capacity(
    model: str, size: str, revision: str, tp_size: int
) -> None:
    config = load_model_config(_config_path(model, tp_size))
    command = build_serve_command(config, host="127.0.0.1", port=8000)
    assert command[4] == model_load_path(config)
    assert command[command.index("--revision") + 1] == revision
    assert command[command.index("--world-size") + 1] == str(tp_size)
    assert command[command.index("--device") + 1] == "npu"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-num-seqs") + 1] == str(config["max_num_seqs"])
    assert config["max_num_seqs"] >= 4
    assert config["max_total_tokens"] >= 4 * config["max_model_len"]
    assert 4 in config["graph_capture_sizes"]
    assert "--disable-prefill-graph" in command
    assert "--disable-pdl" in command
    if model == "9b":
        assert config["hf_overrides"] == {"npu_lm_head_dtype": "float32"}
        assert json.loads(command[command.index("--hf-overrides") + 1]) == {
            "npu_lm_head_dtype": "float32"
        }
    else:
        assert "--hf-overrides" not in command


@pytest.fixture(scope="module")
def mapping_class() -> type:
    # Load the real topology resolver without importing Torch communication code.
    path = ROOT / "python/tokenspeed/runtime/distributed/mapping.py"
    spec = importlib.util.spec_from_file_location("homework2_mapping", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mapping


@pytest.mark.parametrize("model,size,revision", MODELS)
@pytest.mark.parametrize("tp_size", (1, 2))
def test_launch_resolves_one_sharded_model_without_data_parallel_replicas(
    model: str, size: str, revision: str, tp_size: int, mapping_class: type
) -> None:
    config = load_model_config(_config_path(model, tp_size))
    command = build_serve_command(config, host="127.0.0.1", port=8000)
    world_size = int(command[command.index("--world-size") + 1])
    topology_options = {}
    for name in (
        "attn_tp_size",
        "attn_dp_size",
        "dense_tp_size",
        "dense_dp_size",
        "pp_size",
    ):
        flag = f"--{name.replace('_', '-')}"
        if flag in command:
            topology_options[name] = int(command[command.index(flag) + 1])
    assert world_size == tp_size
    for rank in range(world_size):
        mapping = mapping_class(rank=rank, world_size=world_size, **topology_options)
        assert mapping.pp_size == 1
        assert mapping.nnodes == 1
        assert mapping.nprocs_per_node == tp_size
        for component in (mapping.attn, mapping.linear_attn, mapping.dense):
            assert component.tp_size == tp_size
            assert component.tp_rank == rank
            assert component.tp_group == tuple(range(tp_size))
            assert component.dp_size == 1


def test_strategy_changes_only_prefill_communication_granularity():
    base = load_model_config(_config_path("4b", 2))
    candidate = load_model_config(
        ROOT / "configs/homework2/candidates/qwen35_4b_tp2_chunk512.yaml"
    )
    assert {name for name in base if base[name] != candidate[name]} == {
        "chunked_prefill_size"
    }
    assert base["chunked_prefill_size"] == 4096
    assert candidate["chunked_prefill_size"] == 512


def test_capacity_fallback_keeps_matched_tp1_tp2_configuration():
    pairs = [
        load_model_config(
            ROOT / f"configs/homework2/candidates/qwen35_27b_tp{tp}_memory095.yaml"
        )
        for tp in (1, 2)
    ]
    assert {name for name in pairs[0] if pairs[0][name] != pairs[1][name]} == {
        "tp_size"
    }
    for tp, candidate in zip((1, 2), pairs):
        base = load_model_config(_config_path("27b", tp))
        assert {name for name in base if base[name] != candidate[name]} == {
            "memory_utilization"
        }
        assert candidate["memory_utilization"] == 0.95
