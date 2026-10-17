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

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
from test_session_kv import _load_method


@pytest.mark.parametrize("operation", ["get_internal_state", "set_internal_state"])
def test_control_request_receives_first_and_repeated_replies_without_generation(
    operation,
):
    class GetRequest:
        pass

    method = _load_method(
        "python/tokenspeed/runtime/engine/scheduler_control_client.py",
        "SchedulerControlClient",
        operation,
        GetInternalStateReq=GetRequest,
    )

    class Frontend:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.handle_task = None
            self.observed_requests = []
            self.get_internal_state_communicator = self.communicate
            self.set_internal_state_communicator = self.communicate

        def auto_create_handle_loop(self):
            if self.handle_task is None:
                self.handle_task = asyncio.create_task(self.handle_replies())

        async def handle_replies(self):
            while True:
                request, future = await self.incoming.get()
                self.observed_requests.append(request)
                future.set_result(
                    [
                        SimpleNamespace(internal_state={"rank": 0}, updated=True),
                        SimpleNamespace(internal_state={"rank": 1}, updated=False),
                    ]
                )

        async def communicate(self, request):
            future = asyncio.get_running_loop().create_future()
            await self.incoming.put((request, future))
            return await future

    async def run():
        frontend = Frontend()
        update = object()
        arguments = () if operation == "get_internal_state" else (update,)
        expected = [{"rank": 0}, {"rank": 1}] if not arguments else [True, False]
        try:
            for _ in range(2):
                actual = await asyncio.wait_for(
                    method(frontend, *arguments), timeout=0.5
                )
                assert actual == expected
            assert len(frontend.observed_requests) == 2
            if arguments:
                assert all(r is update for r in frontend.observed_requests)
            else:
                assert all(
                    isinstance(r, GetRequest) for r in frontend.observed_requests
                )
        finally:
            if frontend.handle_task is not None:
                frontend.handle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await frontend.handle_task

    asyncio.run(run())
