from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfilerBackend:
    profiler: object
    activity_map: dict[str, object]
    is_npu: bool


def default_profiler_activities(*, is_npu: bool) -> list[str]:
    return ["CPU", "NPU"] if is_npu else ["CPU", "GPU"]


def resolve_profiler_backend(
    *,
    is_npu: bool,
    torch_module: object,
    import_module: Callable[[str], object] = importlib.import_module,
) -> ProfilerBackend:
    if is_npu:
        profiler = getattr(import_module("torch_npu"), "profiler")
        activity = getattr(profiler, "ProfilerActivity")
        return ProfilerBackend(
            profiler=profiler,
            activity_map={"CPU": activity.CPU, "NPU": activity.NPU},
            is_npu=True,
        )

    profiler = getattr(torch_module, "profiler")
    activity = getattr(profiler, "ProfilerActivity")
    return ProfilerBackend(
        profiler=profiler,
        activity_map={"CPU": activity.CPU, "GPU": activity.CUDA},
        is_npu=False,
    )


def create_profiler(
    backend: ProfilerBackend,
    *,
    activities: list[str],
    with_stack: bool,
    record_shapes: bool,
) -> object | None:
    profiler_level = os.environ.get("TOKENSPEED_NPU_PROFILER_LEVEL", "Level0")
    if profiler_level not in {"Level0", "Level1"}:
        raise ValueError(
            "TOKENSPEED_NPU_PROFILER_LEVEL must be Level0 or Level1; "
            f"got {profiler_level!r}"
        )

    experimental_config = None
    if profiler_level == "Level1":
        if not backend.is_npu:
            raise RuntimeError(
                "TOKENSPEED_NPU_PROFILER_LEVEL=Level1 is only supported on NPU"
            )
        try:
            experimental_config = backend.profiler._ExperimentalConfig(
                profiler_level=backend.profiler.ProfilerLevel.Level1,
                aic_metrics=backend.profiler.AiCMetrics.PipeUtilization,
            )
        except Exception as exc:
            raise RuntimeError(
                "Torch-NPU profiler Level1 with PipeUtilization is unsupported: "
                f"{exc}"
            ) from exc

    profiler_activities = [
        backend.activity_map[name]
        for name in activities
        if name in backend.activity_map
    ]
    if not profiler_activities:
        return None

    profile = getattr(backend.profiler, "profile")
    if backend.is_npu:
        kwargs = {
            "activities": profiler_activities,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
            "profile_memory": "MEM" in activities,
        }
        if experimental_config is not None:
            kwargs["experimental_config"] = experimental_config
        return profile(**kwargs)
    return profile(
        activities=profiler_activities,
        with_stack=with_stack,
        record_shapes=record_shapes,
    )


def trace_file_name(stem: str, *, is_npu: bool) -> str:
    suffix = ".trace.json" if is_npu else ".trace.json.gz"
    return f"{stem}{suffix}"
