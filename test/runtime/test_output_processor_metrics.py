from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, generate_latest

from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut
from tokenspeed.runtime.engine.output_processor import OutputProcessor, ReqState
from tokenspeed.runtime.metrics.collector import RequestMetrics


def _batch(*, completion_tokens: int, generated_time: float, finished: bool):
    return BatchTokenIDOut(
        rids=["request"],
        finished_reasons=[{"type": "stop"} if finished else None],
        decoded_texts=[""],
        decode_ids=[[]],
        read_offsets=[0],
        output_ids=[[]],
        output_multi_ids=None,
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[8],
        completion_tokens=[completion_tokens],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[[]],
        output_token_logprobs_idx=[[]],
        input_top_logprobs_val=[[]],
        input_top_logprobs_idx=[[]],
        output_top_logprobs_val=[[]],
        output_top_logprobs_idx=[[]],
        input_token_ids_logprobs_val=[[]],
        input_token_ids_logprobs_idx=[[]],
        output_token_ids_logprobs_val=[[]],
        output_token_ids_logprobs_idx=[[]],
        output_hidden_states=[[]],
        batch_accept_draft_tokens=[],
        output_extra_infos=[{}],
        generated_time=generated_time,
    )


def test_inter_token_histogram_records_engine_intervals_once_per_token():
    registry = CollectorRegistry()
    metrics = RequestMetrics({"model_name": "test"}, enabled=True, registry=registry)
    processor = OutputProcessor(SimpleNamespace(metrics=metrics))
    state = ReqState(
        collector=RequestOutputCollector(),
        finished=False,
        event=asyncio.Event(),
        obj=SimpleNamespace(),
        created_time=0.0,
    )

    processor.collect_metrics(
        state,
        _batch(completion_tokens=1, generated_time=10.0, finished=False),
        0,
    )
    state.finished = True
    state.finished_time = state.first_token_time + 0.3
    processor.collect_metrics(
        state,
        _batch(completion_tokens=4, generated_time=10.12, finished=True),
        0,
    )

    exposition = generate_latest(registry).decode("utf-8")
    count = next(
        float(line.rsplit(" ", 1)[1])
        for line in exposition.splitlines()
        if line.startswith("tokenspeed:inter_token_latency_seconds_count{")
    )
    total = next(
        float(line.rsplit(" ", 1)[1])
        for line in exposition.splitlines()
        if line.startswith("tokenspeed:inter_token_latency_seconds_sum{")
    )

    assert count == 3
    assert math.isclose(total, 0.12, abs_tol=1e-9)
