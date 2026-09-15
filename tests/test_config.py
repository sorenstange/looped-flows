from pathlib import Path

import pytest

from src.config import Config, load_config, save_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_yaml_matches_schema_defaults():
    assert load_config(CONFIGS / "default.yaml") == Config()


def test_extends_and_overrides(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(f"extends: {(CONFIGS / 'smoke.yaml').as_posix()}\noracle:\n  cost: 0.001\n")
    cfg = load_config(child, ["features.lookback=32", "data.end=null"])
    assert cfg.oracle.cost == 0.001  # child
    assert cfg.oracle.horizon == 16  # smoke
    assert cfg.data.symbol == "BTCUSDT"  # default
    assert cfg.features.lookback == 32  # CLI override
    assert cfg.data.end is None


def test_save_and_reload_roundtrip(tmp_path):
    cfg = load_config(CONFIGS / "smoke.yaml", ["oracle.num_levels=5", "oracle.max_step=null",
                                               "oracle_sweep.max_steps=[0.5,null]"])
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == cfg


@pytest.mark.parametrize("override", [
    "data.interval=1x", "oracle.return_type=pct", "oracle.num_levels=1", "oracle.max_step=0.05",
    "features.scaling=zscore", "features.rolling_window=1", "features.clip=0",
    "splits.val_start=2026-01-01", "oracle.cost=-0.1", "inference.readout_step=0", "inference.readout_step=65",
    "inference.aggregation=median", "inference.readout=last", "inference.samples=0", "act.tolerance=-0.1",
    "act.steps=0", "act.steps=65", "act.loss_weight=-1",
])
def test_invalid_values_are_rejected(override):
    with pytest.raises(ValueError):
        load_config(CONFIGS / "default.yaml", [override])


def test_unknown_keys_are_rejected():
    with pytest.raises(Exception):
        load_config(CONFIGS / "default.yaml", ["oracle.horizn=10"])
