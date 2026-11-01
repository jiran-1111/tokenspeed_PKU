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

"""CPU tests for fresh receipts and refusal to trust changed model files."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prepare_reproduction import file_origin, model_state, sha, write_json


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[4]
        self.temporary = tempfile.TemporaryDirectory(dir=repo / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "models/Qwen-test"
        self.model.mkdir(parents=True)
        self.weight = self.model / "weights.bin"
        self.weight.write_bytes(b"fixed model fixture")
        self.expected = [
            {
                "model_id": "Qwen/Qwen-test",
                "model_revision": "fixed-revision",
                "files": {
                    "weights.bin": {
                        "bytes": self.weight.stat().st_size,
                        "sha256": sha(self.weight),
                    }
                },
            }
        ]

    def test_fresh_stat_and_hash_match_current_file(self):
        result = model_state(self.root, self.expected)["models"]["Qwen/Qwen-test"]
        self.assertEqual(result["revision"], "fixed-revision")
        value = self.weight.stat()
        self.assertEqual(
            result["files"]["weights.bin"]["stat"],
            [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns],
        )

    def test_same_size_changed_model_is_rejected(self):
        self.weight.write_bytes(b"changed model text!")
        self.assertEqual(
            self.weight.stat().st_size,
            self.expected[0]["files"]["weights.bin"]["bytes"],
        )
        with self.assertRaisesRegex(ValueError, "model identity mismatch"):
            model_state(self.root, self.expected)

    def test_file_escape_is_rejected(self):
        self.expected[0]["files"] = {
            "../../outside.bin": self.expected[0]["files"]["weights.bin"]
        }
        (self.root / "outside.bin").write_bytes(b"fixed model fixture")
        with self.assertRaisesRegex(ValueError, "escapes"):
            model_state(self.root, self.expected)

    def test_outside_import_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            file_origin(SimpleNamespace(__file__=__file__), self.root)

    def test_existing_receipt_is_not_overwritten(self):
        receipt = self.root / "receipt.json"
        write_json(receipt, {"first": True})
        with self.assertRaises(FileExistsError):
            write_json(receipt, {"second": True})
        self.assertEqual(json.loads(receipt.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()
