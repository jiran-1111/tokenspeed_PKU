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

"""Exercise own-group cleanup without a model or accelerator."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from owned_processes import terminate_owned_group


def start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]


control = subprocess.Popen(
    [sys.executable, "-c", "import time;time.sleep(120)"], start_new_session=True
)
results = []
try:
    for exits_first in (False, True):
        marker = Path(__file__).with_name(f"cleanup-child-{int(exits_first)}.txt")
        code = (
            f"import pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); pathlib.Path({str(marker)!r}).write_text(str(p.pid)); "
            + ("time.sleep(.5)" if exits_first else "time.sleep(120)")
        )
        parent = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        ticks = start_ticks(parent.pid)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        child = int(marker.read_text())
        if exits_first:
            parent.wait(timeout=5)
        terminate_owned_group(parent, expected_start=ticks)
        assert control.poll() is None
        status = Path(f"/proc/{child}/stat")
        assert (
            not status.exists()
            or status.read_text().rpartition(")")[2].split()[0] == "Z"
        )
        results.append(
            {
                "parent_exited_before_cleanup": exits_first,
                "owned_child_stopped": True,
                "unrelated_control_alive": True,
            }
        )
    Path(__file__).with_name("timeout_selftest.json").write_text(
        json.dumps({"passed": True, "cases": results}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"passed": True, "cases": len(results), "unrelated_control_affected": False}
        )
    )
finally:
    if control.poll() is None:
        # The control is the exact untouched Popen object created by this test.
        control.terminate()
        control.wait(timeout=5)
