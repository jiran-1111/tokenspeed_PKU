"""0.8B Engine Session isolation and close/reopen acceptance; not a benchmark."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from benchmarks.course_project.core import load_model_config, model_load_path
from scripts.run_session_kv import _build_engine_config_and_cli_args, _run_sessions

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def test_08b_sessions_close_reopen(record_property) -> None:
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    root = Path(__file__).resolve().parents[2]
    config = load_model_config(root / "configs/baseline/qwen35_0.8b.yaml")
    assert config["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert config["tp_size"] == 1
    config, cli = _build_engine_config_and_cli_args(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path(config),
        revision=config["model_revision"],
        local_files_only=True,
    )
    rounds = []
    record_property("config", json.dumps(config))
    with Engine(server_args=prepare_server_args(cli)) as engine:
        for index in range(2):
            record = {"turns": [], "closed_session_rejections": []}
            try:
                asyncio.run(
                    _run_sessions(
                        engine,
                        tokenizer,
                        frontier_policy="current-frontier",
                        max_new_tokens=8,
                        run_record=record,
                    )
                )
            finally:
                record_property(f"session_round_{index}", json.dumps(record))
            assert len(record["turns"]) == 4
            assert len(record["closed_session_rejections"]) == 2
            assert all(
                turn["output_ids_match"] and turn["context_metadata_validated"]
                for turn in record["turns"]
            )
            rounds.append(record)
        assert set(rounds[0]["session_ids"]).isdisjoint(rounds[1]["session_ids"])
        for original, reopened in zip(rounds[0]["turns"], rounds[1]["turns"]):
            assert original["session_label"] == reopened["session_label"]
            assert original["turn_index"] == reopened["turn_index"]
            assert (
                original["stateless_control_input_ids"]
                == reopened["stateless_control_input_ids"]
            )
            assert original["session_output_ids"] == reopened["session_output_ids"]
