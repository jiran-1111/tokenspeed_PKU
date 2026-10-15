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

import pytest

from scripts import run_homework1_decode_profile as profile


def test_profile_starts_after_first_output_and_stops_during_decode():
    events = []

    class Engine:
        def generate(self, **kwargs):
            assert (
                kwargs["stream"] and kwargs["sampling_params"]["max_new_tokens"] == 256
            )
            previous = 0
            for size in (1, 2, 128, 256):
                events.append(size)
                yield {"output_ids": list(range(previous, size))}
                previous = size

    record = {}
    profile.capture_decode_window(
        Engine(),
        input_ids=[9],
        start_profile=lambda: events.append("start"),
        stop_profile=lambda: events.append("stop"),
        record=record,
    )
    assert events == [1, "start", 2, 128, "stop", 256]
    assert (
        record["started"]
        and record["stopped"]
        and len(record["final"]["output_ids"]) == 256
    )
    assert [row["tokens"] for row in record["chunks"]] == [1, 2, 128, 256]


def test_profile_is_stopped_when_stream_fails():
    events = []

    class Engine:
        def generate(self, **kwargs):
            yield {"output_ids": [1]}
            raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        profile.capture_decode_window(
            Engine(),
            input_ids=[9],
            start_profile=lambda: events.append("start"),
            stop_profile=lambda: events.append("stop"),
            record={},
        )
    assert events == ["start", "stop"]


def test_late_first_chunk_cannot_be_a_valid_decode_window():
    class Engine:
        def generate(self, **kwargs):
            yield {"output_ids": list(range(256))}

    def unexpected():
        raise AssertionError("Profiler must not start after generation")

    with pytest.raises(ValueError, match="capture frontier"):
        profile.capture_decode_window(
            Engine(),
            input_ids=[9],
            start_profile=unexpected,
            stop_profile=unexpected,
            record={},
        )
