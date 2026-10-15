from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def profiler_control_url(api_url: str) -> str:
    parsed = urlsplit(api_url)
    if parsed.hostname is None:
        raise ValueError("api_url must include a host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port >= 65535:
        raise ValueError("api_url port leaves no room for the control server port")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit(
        parsed._replace(
            netloc=f"{host}:{port + 1}",
            path=path,
            query="",
            fragment="",
        )
    )


class ProfilerControlClient:
    def __init__(self, server_url: str, *, timeout_seconds: float = 300) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def start(self, *, output_dir: Path, profile_id: str) -> dict:
        return self._post(
            "/start_profile",
            {
                "output_dir": str(output_dir),
                "activities": ["CPU", "NPU", "MEM"],
                "with_stack": False,
                "record_shapes": True,
                "profile_id": profile_id,
            },
        )

    def stop(self) -> dict:
        return self._post("/stop_profile", {})

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read() or b"{}")
        if result.get("success") is False:
            raise RuntimeError(result.get("message", "Profiler request failed"))
        return result
