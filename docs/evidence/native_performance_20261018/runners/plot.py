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

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    datasets = {}
    provenance = {}
    labels = {
        "0.8b": ("Qwen/Qwen3.5-0.8B", "TP1"),
        "4b": ("Qwen/Qwen3.5-4B", "TP1"),
        "9b": ("Qwen/Qwen3.5-9B", "TP1, FP32 head"),
        "27b": ("Qwen/Qwen3.5-27B", "TP2"),
    }
    for model, (model_id, _) in labels.items():
        audit_path = a.input_root / model / "audit.json"
        csv_path = a.input_root / model / "requests.csv"
        audit = json.loads(audit_path.read_text())
        assert audit["protocol_and_quality_passed"] and audit["formal_requests"] == 192
        assert (
            audit["source_manifest_sha256"] == a.source_manifest_sha256
            and audit["trace_sha256"] == protocol["traces"][model_id]["sha256"]
        )
        with csv_path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["phase"] == "formal"]
        assert len(rows) == 192 and {
            (
                int(r["repetition"]),
                r["mode"],
                int(r["session_index"]),
                int(r["turn_index"]),
            )
            for r in rows
        } == {
            (rep, mode, index, turn)
            for rep in range(3)
            for mode in ("session", "stateless")
            for index in range(2)
            for turn in range(1, 17)
        }
        datasets[model] = rows
        provenance[model] = {
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "requests_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "run_sha256": audit["run_sha256"],
            "trace_sha256": audit["trace_sha256"],
        }
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7))
    for ax, (model, rows) in zip(axes.flat, datasets.items()):
        for mode, label, color in (
            ("session", "Session ON", "#0072B2"),
            ("stateless", "Session OFF", "#D55E00"),
        ):
            values = [
                [
                    float(r["ttft_ms"])
                    for r in rows
                    if r["mode"] == mode and int(r["turn_index"]) == turn
                ]
                for turn in range(1, 17)
            ]
            assert all(len(v) == 6 and all(x > 0 for x in v) for v in values)
            ax.plot(
                range(1, 17),
                [statistics.median(v) for v in values],
                label=label,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(
                range(1, 17),
                [min(v) for v in values],
                [max(v) for v in values],
                color=color,
                alpha=0.13,
                linewidth=0,
            )
        ax.set_title(f'{labels[model][0].split("/")[-1]} ({labels[model][1]})')
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_xlabel("Conversation turn")
        ax.set_ylabel("Client TTFT (ms, log scale)")
        ax.grid(True, which="major", alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Fixed native Agent trajectory: independent Session ON/OFF arms", fontsize=13
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.12, top=0.88, hspace=0.4, wspace=0.28
    )
    fig.text(
        0.5,
        0.035,
        "Median line; min-max band over 2 sessions x 3 repeats at each turn. Shared-engine repeated observations; not confidence intervals.",
        ha="center",
        fontsize=8,
    )
    for suffix in ("svg", "png"):
        fig.savefig(a.output_dir / f"ttft_by_turn.{suffix}", dpi=200)
    plt.close(fig)
    (a.output_dir / "figure_provenance.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": a.source_manifest_sha256,
                "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
                "inputs": provenance,
                "interpretation": "fixed real-trajectory conditional replay, 3 alternating-order repetitions within one engine; warmups excluded; bands descriptive, not CIs",
            },
            indent=2,
        )
        + "\n"
    )
    (a.output_dir / "figure_caption.md").write_text(
        "Client TTFT by conversation turn for fixed R8 trajectories. Each arm runs two concurrent sessions for 16 turns with eight output tokens per turn and the scripted two-second tool wait. Lines show medians and bands show min/max of the six session/repetition observations per turn. Warmups are excluded. The log scale is explicit; bands are descriptive ranges rather than confidence intervals. 9B uses the explicit FP32 output-head candidate, and 27B uses TP2.\n"
    )
    print(
        json.dumps(
            {
                "models": 4,
                "formal_requests": 768,
                "figures": ["ttft_by_turn.svg", "ttft_by_turn.png"],
            }
        )
    )


if __name__ == "__main__":
    main()
