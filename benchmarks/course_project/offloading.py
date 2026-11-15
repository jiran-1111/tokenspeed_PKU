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

"""Summarize actual per-rank cache traces without inferring HBM release from RSS."""

from __future__ import annotations

import json
from pathlib import Path


def summarize_run(directory: Path) -> dict:
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    ranks = []
    for path in sorted((directory / "trace").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        snapshots = [row for row in rows if row["kind"] == "scheduler"]
        if not snapshots:
            continue
        allocation = next((r for r in rows if r["kind"] == "host_allocation"), {})
        totals = {
            "rank": snapshots[0]["rank"],
            "host_allocated_bytes": allocation.get("allocated_bytes", 0),
        }
        owners = {turn["meta_info"]["id"]: turn["session"] for turn in result["turns"]}
        block_owners = {}
        reused = set()
        restored_positions = []
        for snapshot in snapshots:
            restored_positions.extend(snapshot["restore_positions"])
            for group, tables in snapshot["block_tables"].items():
                for rid, table in zip(snapshot["request_ids"], tables):
                    owner = owners.get(rid)
                    if owner is None:
                        continue
                    for block in table:
                        if block <= 0:
                            continue
                        key = (group, block)
                        if key in block_owners and block_owners[key] != owner:
                            reused.add(key)
                        block_owners[key] = owner
        totals["reallocated_blocks_between_sessions"] = len(reused)
        totals["restore_positions"] = sorted(set(restored_positions))
        for key in (
            "device_active_blocks",
            "device_available_blocks",
            "host_cached_blocks",
            "host_pinned_blocks",
            "host_hits",
            "host_misses",
            "host_evictions",
            "idle_session_reclaims",
        ):
            totals[f"peak_{key}" if key.endswith("blocks") else key] = max(
                row[key] for row in snapshots
            )
        for direction in ("d2h", "h2d"):
            transfers = [
                r
                for r in rows
                if r["kind"] == "transfer" and r["direction"] == direction
            ]
            totals[f"{direction}_bytes"] = sum(r["payload_bytes"] for r in transfers)
            totals[f"{direction}_ms"] = sum(r["duration_ms"] for r in transfers)
        # Exclude logical null parent 0. Occupied parents include pinned transfers.
        if allocation:
            totals["peak_host_occupied_bytes"] = max(
                allocation["allocated_bytes"]
                - row["host_free_blocks"] * allocation["lcm_block_bytes"]
                for row in snapshots
            )
        ranks.append(totals)
    measured = [turn for turn in result["turns"] if turn["cycle"] > 0]
    return {
        "status": result["status"],
        "turns": len(result["turns"]),
        "matched_controls": sum(c["match"] for c in result["controls"]),
        "ranks": ranks,
        "measured_turns": len(measured),
        "request_time_output_tps": (
            (
                sum(len(t["output_ids"]) for t in measured)
                / (sum(t["elapsed_ms"] for t in measured) / 1000)
            )
            if measured
            else None
        ),
        "offload_restore_observed": bool(ranks)
        and all(
            r["idle_session_reclaims"] > 0 and r["d2h_bytes"] > 0 and r["h2d_bytes"] > 0
            for r in ranks
        ),
    }
