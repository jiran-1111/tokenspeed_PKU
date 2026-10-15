from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.course_project.profiler import ProfilerControlClient


class TestProfilerControlClient(unittest.TestCase):
    def test_default_timeout_allows_npu_trace_export(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with patch(
            "benchmarks.course_project.profiler.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            result = ProfilerControlClient("http://127.0.0.1:8000").stop()

        self.assertEqual(result, {})
        self.assertEqual(open_url.call_args.kwargs["timeout"], 300)

    def test_start_raises_with_message_when_profiler_rejects_request(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"success": false, "message": "capture was rejected"}'
        )
        with patch(
            "benchmarks.course_project.profiler.urllib.request.urlopen",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture was rejected"):
                ProfilerControlClient("http://127.0.0.1:8000").start(
                    output_dir=Path("profiles"),
                    profile_id="profile-1",
                )


if __name__ == "__main__":
    unittest.main()
