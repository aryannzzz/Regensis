from __future__ import annotations

from regenesis_cmi.cli import _parser
from regenesis_cmi.evaluation.benchmark import estimate_runtime, load_config
import pandas as pd


def test_cli_and_all_configs_import_and_parse() -> None:
    parser = _parser()
    assert parser.prog == "regenesis-cmi"
    for path in ("configs/smoke.yaml", "configs/primary.yaml", "configs/extended.yaml"):
        config = load_config(path)
        estimate = estimate_runtime(config)
        assert estimate["gpu_required"] is False


def test_csv_policy_preserves_null_world_label(tmp_path) -> None:
    path = tmp_path / "world.csv"
    path.write_text("world,value\nnull,0\n", encoding="utf-8")
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    assert frame.loc[0, "world"] == "null"
