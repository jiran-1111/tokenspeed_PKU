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

"""Opt-in per-process cache evidence for capacity and transfer experiments."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class CacheTrace:
    def __init__(self) -> None:
        directory = os.environ.get("TOKENSPEED_CACHE_TRACE_DIR")
        self.path = (
            None if not directory else Path(directory) / f"cache-{os.getpid()}.jsonl"
        )
        self.pending: list[tuple] = []
        self.lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **values) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"kind": kind, "time_ns": time.time_ns(), **values}) + "\n"
            )

    def start_transfer(self, device_module, stream):
        if self.path is None:
            return None
        event = device_module.Event(enable_timing=True)
        event.record(stream)
        return event

    def track_transfer(
        self, start, finish, direction, transfers, host_block_bytes
    ) -> None:
        if start is not None:
            payload_bytes = sum(host_block_bytes[group] for group, _, _ in transfers)
            with self.lock:
                self.pending.append(
                    (start, finish, direction, payload_bytes, list(transfers))
                )

    def poll(self) -> None:
        with self.lock:
            self._poll_locked()

    def _poll_locked(self) -> None:
        remaining = []
        for start, finish, direction, payload_bytes, transfers in self.pending:
            if finish.query():
                self.write(
                    "transfer",
                    direction=direction,
                    payload_bytes=payload_bytes,
                    duration_ms=start.elapsed_time(finish),
                    blocks=transfers,
                )
            else:
                remaining.append((start, finish, direction, payload_bytes, transfers))
        self.pending = remaining
