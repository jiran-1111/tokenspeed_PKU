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

import math

import pytest

from scripts.analyze_homework1_stage_b import GATES, gate_result


@pytest.mark.parametrize("step", [0, 1, 16])
def test_frozen_boundary_and_each_failure(step):
    l2, absolute = GATES["prefill" if step == 0 else "decode"]
    metrics = {"relative_l2": l2, "max_abs": absolute, "finite": True}
    assert gate_result(metrics, step=step)["frozen_pass"]
    assert not gate_result(metrics, step=step)["original_1pct_pass"]
    assert not gate_result(
        {**metrics, "relative_l2": math.nextafter(l2, math.inf)}, step=step
    )["frozen_pass"]
    assert not gate_result(
        {**metrics, "max_abs": math.nextafter(absolute, math.inf)}, step=step
    )["frozen_pass"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -1])
def test_invalid_metrics_rejected(value):
    with pytest.raises(ValueError):
        gate_result({"relative_l2": value, "max_abs": 0, "finite": True}, step=0)
