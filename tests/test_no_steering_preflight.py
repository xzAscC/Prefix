from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import cast

import pytest

from prefix import no_steering


SCRIPT = Path(__file__).parents[1] / "scripts" / "no_steering_preflight.py"
MODEL_IDS = tuple(no_steering.MODEL_MATRIX)
FULL_LOAD_IDS = ("Qwen/Qwen3-4B", "allenai/Olmo-3-7B-Think")
PROTOCOL_ONLY_IDS = ("Qwen/Qwen3-14B", "allenai/Olmo-3-32B-Think")


def entrypoint():
    if not SCRIPT.exists():
        pytest.fail(f"missing local no-steering preflight entrypoint: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("no_steering_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_full_load_and_protocol_only_boundaries_are_explicit() -> None:
    preflight = entrypoint()
    assert tuple(preflight.LOCAL_FULL_LOAD_MODEL_IDS) == FULL_LOAD_IDS
    assert tuple(preflight.PROTOCOL_ONLY_MODEL_IDS) == PROTOCOL_ONLY_IDS
    assert set(preflight.LOCAL_FULL_LOAD_MODEL_IDS).isdisjoint(
        preflight.PROTOCOL_ONLY_MODEL_IDS
    )
    assert set(preflight.LOCAL_FULL_LOAD_MODEL_IDS) | set(
        preflight.PROTOCOL_ONLY_MODEL_IDS
    ) == set(MODEL_IDS)
    for model_id in tuple(preflight.LOCAL_FULL_LOAD_MODEL_IDS):
        assert preflight.model_spec(model_id) == no_steering.model_spec(model_id)


def test_missing_snapshot_is_rejected_before_any_cuda_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = entrypoint()
    cuda_calls: list[str] = []

    class FakeCuda:
        def is_available(self) -> bool:
            cuda_calls.append("is_available")
            return True

    monkeypatch.setattr(preflight, "torch", type("Torch", (), {"cuda": FakeCuda()})())
    with pytest.raises(FileNotFoundError, match="Qwen/Qwen3-4B"):
        preflight.check_model_snapshots(
            MODEL_IDS,
            cache_root=tmp_path,
        )
    assert cuda_calls == []


def test_existing_local_snapshots_are_checked_without_loading_or_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = entrypoint()
    snapshot_paths = {
        model_id: _write_snapshot(tmp_path, model_id) for model_id in MODEL_IDS
    }
    load_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        preflight,
        "load_model",
        lambda **kwargs: load_calls.append(kwargs) or object(),
    )

    result = preflight.check_model_snapshots(
        MODEL_IDS,
        cache_root=tmp_path,
        load=False,
    )

    assert result == snapshot_paths
    assert load_calls == []


def test_symlinked_snapshot_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    real_snapshot = _write_snapshot(tmp_path, model_id)
    snapshot = preflight._snapshot_path(model_id, tmp_path)
    real_target = real_snapshot.parent / "real-target"
    real_snapshot.rename(real_target)
    snapshot.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="exact pinned revision"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


@pytest.mark.parametrize(
    "asset", ["config.json", "tokenizer.json", "tokenizer_config.json"]
)
def test_rejects_empty_required_asset_link(tmp_path: Path, asset: str) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id)
    target = snapshot.parent.parent / f"empty-{asset}"
    target.touch()
    (snapshot / asset).unlink()
    (snapshot / asset).symlink_to(target)

    with pytest.raises((ValueError, FileNotFoundError), match="nonempty|invalid"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


@pytest.mark.parametrize("target_kind", ["dangling", "escaping"])
def test_rejects_unsafe_weight_links(tmp_path: Path, target_kind: str) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    target = (
        tmp_path / "missing-weight"
        if target_kind == "dangling"
        else tmp_path.parent / "escaping-weight"
    )
    (snapshot / "model.safetensors").symlink_to(target)

    with pytest.raises((FileNotFoundError, ValueError), match="weight|repo|symlink"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


def test_full_local_load_uses_only_two_models_and_forbids_steering_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = entrypoint()
    loaded: list[dict[str, object]] = []
    monkeypatch.setattr(
        preflight,
        "load_model",
        lambda **kwargs: loaded.append(kwargs) or object(),
    )

    preflight.load_local_models(
        cache_root=tmp_path,
        model_ids=MODEL_IDS,
        snapshots={model_id: tmp_path for model_id in MODEL_IDS},
    )

    assert [call["model_id"] for call in loaded] == list(FULL_LOAD_IDS)
    assert all(call["steering"] is False for call in loaded)
    assert all(call["quantization"] in (None, False) for call in loaded)
    assert all(
        call["revision"] == no_steering.model_spec(cast(str, call["model_id"])).revision
        for call in loaded
    )


def _write_snapshot(
    cache_root: Path,
    model_id: str,
    *,
    revision: str | None = None,
    quantized: bool = False,
    broken_blob: bool = False,
    weight_name: str | None = "model.safetensors",
) -> Path:
    spec = no_steering.model_spec(model_id)
    revision = revision or spec.revision
    repo = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(exist_ok=True)
    config: dict[str, object] = {"model_type": "synthetic"}
    if quantized:
        config["quantization_config"] = {"bits": 4}
    (blobs / "config").write_text(json.dumps(config))
    (blobs / "tokenizer").write_text("{}")
    (snapshot / "config.json").symlink_to(blobs / "config")
    (snapshot / "tokenizer.json").symlink_to(
        blobs / ("missing-tokenizer" if broken_blob else "tokenizer")
    )
    (snapshot / "tokenizer_config.json").write_text("{}")
    if weight_name is not None:
        (snapshot / weight_name).write_bytes(b"synthetic weights")
    return snapshot


def test_resolves_exact_hf_snapshot_and_writes_atomic_revision_manifest(
    tmp_path: Path,
) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshots = {
        current_id: _write_snapshot(tmp_path, current_id) for current_id in MODEL_IDS
    }
    manifest = tmp_path / "assets.json"

    result = preflight.check_model_snapshots(
        MODEL_IDS, cache_root=tmp_path, load=False, manifest_path=manifest
    )

    assert result == snapshots
    payload = json.loads(manifest.read_text())
    assert set(payload["models"]) == set(MODEL_IDS)
    assert (
        payload["models"][model_id]["revision"]
        == no_steering.model_spec(model_id).revision
    )
    assert payload["models"][model_id]["snapshot"] == str(snapshots[model_id])
    assert not list(tmp_path.glob("assets.json.tmp.*"))


def test_atomic_preflight_manifest_rejects_symlinked_destination_parent(
    tmp_path: Path,
) -> None:
    preflight = entrypoint()
    for model_id in MODEL_IDS:
        _write_snapshot(tmp_path, model_id)
    victim = tmp_path / "victim.json"
    victim.write_text("safe", encoding="utf-8")
    manifest = tmp_path / "assets.json"
    manifest.symlink_to(victim)

    with pytest.raises(ValueError, match="symlink"):
        preflight.check_model_snapshots(
            MODEL_IDS, cache_root=tmp_path, load=False, manifest_path=manifest
        )
    assert victim.read_text(encoding="utf-8") == "safe"


def test_manifest_requires_the_exact_four_model_set(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    _write_snapshot(tmp_path, model_id)

    with pytest.raises(ValueError, match="exact required models"):
        preflight.check_model_snapshots(
            [model_id],
            cache_root=tmp_path,
            load=False,
            manifest_path=tmp_path / "assets.json",
        )


def test_marker_requires_the_exact_four_model_set(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    _write_snapshot(tmp_path, model_id)

    with pytest.raises(ValueError, match="exact required models"):
        preflight.check_model_snapshots(
            [model_id],
            cache_root=tmp_path,
            load=False,
            marker_path=tmp_path / "preflight.ok",
        )


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
def test_accepts_nonempty_direct_weight_files(tmp_path: Path, weight_name: str) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / weight_name).write_bytes(b"weights")

    assert (
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)[
            model_id
        ]
        == snapshot
    )


def test_accepts_safetensors_index_and_all_nonempty_shards(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}})
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    assert (
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)[
            model_id
        ]
        == snapshot
    )


def test_rejects_shard_like_weights_without_an_index(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / "model-00001-of-00010.safetensors").write_bytes(b"partial")

    with pytest.raises(ValueError, match="shard.*index|incomplete"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


@pytest.mark.parametrize(
    "index_payload",
    [{}, {"weight_map": {}}, {"weight_map": {"layer": "missing.safetensors"}}],
)
def test_rejects_empty_or_partial_weight_indexes(
    tmp_path: Path, index_payload: dict[str, object]
) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index_payload))

    with pytest.raises((ValueError, FileNotFoundError), match="weight|shard"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


def test_rejects_empty_weight_and_shard_files(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / "model.safetensors").touch()
    with pytest.raises(ValueError, match="nonempty"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)

    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "empty.safetensors"}})
    )
    (snapshot / "empty.safetensors").touch()
    with pytest.raises(ValueError, match="nonempty"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


def test_rejects_index_shard_outside_cache_repository(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    snapshot = _write_snapshot(tmp_path, model_id, weight_name=None)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../../../../outside.safetensors"}})
    )

    with pytest.raises(ValueError, match="escapes cache repo"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


def test_rejects_wrong_revision_and_dangling_blob(tmp_path: Path) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    _write_snapshot(tmp_path, model_id, revision="0" * 40)
    with pytest.raises(FileNotFoundError, match="exact pinned revision"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)

    _write_snapshot(tmp_path, model_id, broken_blob=True)
    with pytest.raises(FileNotFoundError, match="broken symlink|blob"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)


def test_rejects_quantized_config_and_sets_offline_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = entrypoint()
    model_id = MODEL_IDS[0]
    _write_snapshot(tmp_path, model_id, quantized=True)
    with pytest.raises(ValueError, match="quantization"):
        preflight.check_model_snapshots([model_id], cache_root=tmp_path, load=False)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
