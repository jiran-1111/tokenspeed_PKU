from __future__ import annotations

from copy import deepcopy


def configure_ablation_ports(
    config: dict, base_config: dict, variant_index: int
) -> None:
    config["prometheus_port"] = base_config["prometheus_port"] + variant_index
    config["tokenspeed_metrics_port"] = (
        base_config["tokenspeed_metrics_port"] + variant_index
    )


def build_ablation_configs(
    config: dict, *, chunked_prefill_size: int | None = None
) -> list[dict]:
    """Return upstream CLI ablations while keeping continuous batching enabled."""
    baseline = deepcopy(config)
    baseline.update(
        {
            "chunked_prefill_size": config.get(
                "max_model_len", config["chunked_prefill_size"]
            ),
            "prefix_caching": False,
            "disable_kvstore": True,
            "mixed_batch": False,
            "overlap_schedule": False,
        }
    )

    scheduler = deepcopy(baseline)
    scheduler["overlap_schedule"] = True

    chunked_prefill = deepcopy(scheduler)
    chunked_prefill["chunked_prefill_size"] = (
        config["chunked_prefill_size"]
        if chunked_prefill_size is None
        else chunked_prefill_size
    )

    prefix_cache = deepcopy(chunked_prefill)
    prefix_cache["prefix_caching"] = True

    return [
        {"name": "E0", "label": "TokenSpeed baseline", "config": baseline},
        {
            "name": "E1",
            "label": "overlap scheduling",
            "config": scheduler,
        },
        {
            "name": "E2",
            "label": "chunked prefill",
            "config": chunked_prefill,
        },
        {"name": "E3", "label": "prefix cache", "config": prefix_cache},
    ]
