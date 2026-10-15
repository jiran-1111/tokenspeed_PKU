from pathlib import Path

import pytest

from benchmarks.course_project.ablation import build_ablation_configs
from benchmarks.course_project.core import load_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("0.8b", "4b", "9b", "27b")
BASELINE_CONFIG_DIR = REPO_ROOT / "configs" / "baseline"


def test_baseline_directory_contains_exactly_the_requested_models() -> None:
    expected = {f"qwen35_{model}.yaml" for model in MODELS}
    actual = {path.name for path in BASELINE_CONFIG_DIR.glob("*.yaml")}

    assert actual == expected


@pytest.mark.parametrize("model", MODELS)
def test_baseline_config_matches_derived_e0(model: str) -> None:
    bringup = load_model_config(
        REPO_ROOT / "configs" / "bringup" / f"qwen35_{model}.yaml"
    )
    baseline = load_model_config(BASELINE_CONFIG_DIR / f"qwen35_{model}.yaml")

    expected = build_ablation_configs(bringup)[0]["config"]
    expected.pop("status", None)

    assert baseline == expected
