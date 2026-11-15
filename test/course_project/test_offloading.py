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

"""Cache evidence must describe completed transfers and actual rank snapshots."""

import json
from unittest.mock import Mock

from benchmarks.course_project.offloading import summarize_run
from tokenspeed.runtime.cache.telemetry import CacheTrace


def test_transfer_trace_waits_for_completion_and_counts_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_CACHE_TRACE_DIR", str(tmp_path))
    trace = CacheTrace()
    start, finish = Mock(), Mock()
    start.elapsed_time.return_value = 2.5
    finish.query.return_value = False
    trace.track_transfer(start, finish, "d2h", [(0, 7, 1), (1, 4, 2)], [64, 128])
    trace.poll()
    assert not trace.path.exists()
    finish.query.return_value = True
    trace.poll()
    trace.poll()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["payload_bytes"] == 192
    assert rows[0]["duration_ms"] == 2.5
    assert rows[0]["blocks"] == [[0, 7, 1], [1, 4, 2]]


def test_disabled_trace_never_allocates_device_events(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CACHE_TRACE_DIR", raising=False)
    trace = CacheTrace()
    module = Mock()
    assert trace.start_transfer(module, Mock()) is None
    module.Event.assert_not_called()


def test_output_match_without_transfer_evidence_is_not_offload_validation(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"status": "passed", "turns": [], "controls": []})
    )
    summary = summarize_run(tmp_path)
    assert not summary["offload_restore_observed"]
    assert summary["ranks"] == []


def test_rank_summary_requires_completed_restore_and_tracks_block_reuse(tmp_path):
    result = {
        "status": "passed",
        "controls": [{"match": True}],
        "turns": [
            {
                "cycle": 1,
                "session": "a",
                "meta_info": {"id": "r1"},
                "output_ids": [4],
                "elapsed_ms": 20,
            },
            {
                "cycle": 1,
                "session": "b",
                "meta_info": {"id": "r2"},
                "output_ids": [5],
                "elapsed_ms": 20,
            },
        ],
    }
    (tmp_path / "results.json").write_text(json.dumps(result))
    trace = tmp_path / "trace"
    trace.mkdir()
    rows = [{"kind": "host_allocation", "allocated_bytes": 512, "lcm_block_bytes": 128}]
    for rid, pos in (("r1", 0), ("r2", 128)):
        rows.append(
            {
                "kind": "scheduler",
                "rank": 0,
                "request_ids": [rid],
                "restore_positions": [pos],
                "block_tables": {"kv": [[1, 2]]},
                "device_active_blocks": 2,
                "device_available_blocks": 1,
                "host_cached_blocks": 1,
                "host_pinned_blocks": 0,
                "host_free_blocks": 2,
                "host_hits": 1,
                "host_misses": 1,
                "host_evictions": 0,
                "idle_session_reclaims": 1,
            }
        )
    for direction in ("d2h", "h2d"):
        rows.append(
            {
                "kind": "transfer",
                "direction": direction,
                "payload_bytes": 64,
                "duration_ms": 2.0,
            }
        )
    (trace / "cache-1.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    summary = summarize_run(tmp_path)
    assert summary["offload_restore_observed"]
    rank = summary["ranks"][0]
    assert rank["reallocated_blocks_between_sessions"] == 2
    assert rank["peak_host_occupied_bytes"] == 256
    assert rank["restore_positions"] == [0, 128]
    assert summary["request_time_output_tps"] == 50
