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

"""Task-owned timeout cleanup reused unchanged from frozen HW2 driver."""

import os
import signal
import subprocess
import time
from pathlib import Path


def terminate_owned_group(process: subprocess.Popen, *, expected_start: str) -> None:
    """Signal only our still-running leader's original process group."""

    def members() -> list[int]:
        owned = []
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = path.read_text().rpartition(")")[2].split()
                if int(stat[2]) != process.pid or stat[0] == "Z":
                    continue
                if (
                    int(stat[3]) != process.pid
                    or int(stat[19]) < int(expected_start)
                    or path.stat().st_uid != os.getuid()
                ):
                    raise RuntimeError("child group ownership changed; refusing signal")
                if path.parent.name == str(process.pid) and stat[19] != expected_start:
                    raise RuntimeError("child PID was reused; refusing signal")
                owned.append(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return owned

    for kind, grace in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 10),
    ):
        if not members():
            process.poll()
            return
        try:
            os.killpg(process.pid, kind)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not members():
                process.poll()
                return
            time.sleep(0.2)
    raise RuntimeError("our timed-out experiment could not be terminated")
