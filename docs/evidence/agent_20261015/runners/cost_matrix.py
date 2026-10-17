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

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

repo = Path.cwd()
out = repo / "results/raw/reproduce_9b_head_cost"
out.mkdir(exist_ok=False)
shutil.copyfile(__file__, out / "orchestrator_source.py")
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "2"
orders = [
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
]
(out / "orders.json").write_text(json.dumps(orders) + "\n")
for repetition, order in enumerate(orders):
    for mode in order:
        proc = subprocess.check_output(
            ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
        )
        assert re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", proc)
        with (out / f"r{repetition}_{mode}_driver.log").open("x") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "docs/evidence/agent_20261015/runners/cost_arm.py",
                    mode,
                    str(repetition),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        (out / f"r{repetition}_{mode}_exit.json").write_text(
            json.dumps({"exit_code": result.returncode}) + "\n"
        )
        assert (
            result.returncode == 0
        ), f"Failed {repetition}/{mode}; no automatic restart"
        deadline = time.monotonic() + 30
        while True:
            proc = subprocess.check_output(
                ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
            )
            if re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", proc):
                break
            assert (
                time.monotonic() < deadline
            ), "Target still occupied; do not touch other processes"
            time.sleep(1)
        (out / f"r{repetition}_{mode}" / "post_exit_processes.txt").write_text(proc)
        print(f"Completed arm {repetition}/{mode}; target chip0 free", flush=True)
(out / "result.json").write_text(
    json.dumps(
        {
            "status": "collected",
            "arms": 6,
            "formal_requests": 96,
            "source_manifest": "docs/evidence/agent_20261015/source_manifest.json",
        }
    )
    + "\n"
)
