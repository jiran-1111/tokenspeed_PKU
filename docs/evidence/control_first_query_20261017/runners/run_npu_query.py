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
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

repo = Path.cwd()
raw = repo / "results/raw/reproduce_control_first_query_20261017"
raw.mkdir(parents=True, exist_ok=False)
with (raw / "npu_driver.log").open("x") as log:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "docs/evidence/control_first_query_20261017/runners/npu_query.py",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        timed_out = True
        command = Path(f"/proc/{proc.pid}/cmdline").read_bytes().replace(b"\0", b" ")
        assert (
            b"docs/evidence/control_first_query_20261017/runners/npu_query.py"
            in command
        )
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
(raw / "npu_exit.json").write_text(
    json.dumps(
        {"exit_code": proc.returncode, "timed_out": timed_out, "timeout_seconds": 240}
    )
    + "\n"
)
assert proc.returncode == 0 and not timed_out
deadline = time.monotonic() + 30
while True:
    processes = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "1"], text=True
    )
    if re.search(r"No process in device\.\s*Chip ID\s*:\s*0(?:\s|$)", processes):
        break
    assert (
        time.monotonic() < deadline
    ), "Target occupied after own exit; do not touch other tasks"
    time.sleep(1)
(raw / "post_exit_processes.txt").write_text(processes)
print(
    json.dumps(
        {
            "status": "passed",
            "first_queries_before_generate": 2,
            "normal_generate": True,
            "target_chip0_free": True,
        }
    ),
    flush=True,
)
