from pathlib import Path

import yaml

from prefix.attention_bounds import kv_head_index

ROOT = Path(__file__).parents[1]
MODEL = "allenai/Olmo-3-7B-Think"
REVISION = "d97e442d7cc678210054dbcc9b440894d62c89a4"


def test_paper_experiment_configs_target_olmo3_7b():
    for name in ("exp1", "exp2", "exp3", "exp4"):
        config = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        assert config["model"]["id"] == MODEL
    assert yaml.safe_load((ROOT / "configs/exp1.yaml").read_text())["model"]["layer"] == 17
    assert yaml.safe_load((ROOT / "configs/exp2.yaml").read_text())["model"]["layer"] == 17
    for name in ("exp3", "exp4"):
        config = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        assert max(config["grid"]["layers"]) < 32


def test_section4_config_pins_paper_model_at_displayed_layer_27():
    config = yaml.safe_load((ROOT / "configs/section4_native.yaml").read_text())
    assert config["model"] == MODEL
    assert config["revision"] == REVISION
    assert config["layer"] == 26


def test_query_to_kv_head_mapping_handles_mha_and_gqa():
    assert kv_head_index(16, num_attention_heads=32, num_key_value_heads=32) == 16
    assert kv_head_index(16, num_attention_heads=32, num_key_value_heads=8) == 4
