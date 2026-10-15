"""Exercise capacity seams without importing GPU-dependent recipe modules."""

import ast
from pathlib import Path
from types import SimpleNamespace


def recipe_class():
    root = Path(__file__).resolve().parents[2]
    folder = root / "python/tokenspeed/runtime/layers/attention/kv_cache/recipes"
    methods = {}
    for filename, class_name in (
        ("base.py", "CacheRecipe"),
        ("qwen35.py", "QwenGDNRecipe"),
    ):
        tree = ast.parse((folder / filename).read_text(encoding="utf-8"))
        cls = next(
            x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name
        )
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "num_lcm_blocks",
                "token_capacity",
                "_budgeted_parents",
                "_capped_parents",
                "_capacity_from_parents",
                "_max_packing",
            ):
                node.decorator_list = []
                methods[node.name] = node
    cls = ast.ClassDef(
        name="Recipe",
        bases=[],
        keywords=[],
        body=list(methods.values()),
        decorator_list=[],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {"CacheLayout": object}
    exec(compile(tree, "capacity-seams", "exec"), scope)
    return scope["Recipe"]


def test_qwen_capacity_counts_per_group_parent_demand():
    recipe = recipe_class()()
    recipe.cache_budget_bytes = 100100
    recipe.workspace_bytes = lambda: 100
    recipe.token_limit = 16384
    recipe._shard_counts = {"full": 1, "state": 1}
    # Model a packed hybrid: full-attention packing cannot stand in for
    # recurrent-state working-set demand.
    layout = SimpleNamespace(
        lcm_block_bytes=100,
        prefix_granularity=128,
        group_packing=(("full", 12), ("state", 1)),
    )
    recipe.parents_needed = lambda _, tokens: (tokens + 1535) // 1536 + 32
    parents = recipe.num_lcm_blocks(layout)
    assert parents >= recipe.parents_needed(layout, 16384)
    assert parents <= 999
    assert recipe.token_capacity(layout, parents) == 16384


def test_qwen_capacity_respects_physical_budget():
    recipe = recipe_class()()
    recipe.cache_budget_bytes = 4100
    recipe.workspace_bytes = lambda: 0
    recipe.token_limit = 16384
    recipe._shard_counts = {"full": 1}
    layout = SimpleNamespace(
        lcm_block_bytes=100, prefix_granularity=128, group_packing=(("full", 12),)
    )
    recipe.parents_needed = lambda _, tokens: (tokens + 1535) // 1536 + 32
    parents = recipe.num_lcm_blocks(layout)
    assert parents <= 40
    capacity = recipe.token_capacity(layout, parents)
    assert 0 < capacity < 16384
    assert recipe.parents_needed(layout, capacity) <= parents
