from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from prefix.no_steering import model_spec


ROOT = Path(__file__).parents[1]
MODEL_ID = "Qwen/Qwen3-4B"


def test_failed_smoke_output_rejects_marker_gate(tmp_path: Path) -> None:
    result_root = tmp_path / "Qwen--Qwen3-4B"
    for benchmark in ("harmbench", "mmlu_pro", "math500"):
        path = result_root / benchmark / "responses.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "id": f"{benchmark}-1",
                    "benchmark": benchmark,
                    "model_id": MODEL_ID,
                    "status": "error",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    result_root.mkdir(parents=True, exist_ok=True)
    result_root.joinpath("summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model_id": MODEL_ID,
                    "revision": model_spec(MODEL_ID).revision,
                },
                "ppl": {
                    "ppl": 1.5,
                    "selected_token_count": 6,
                    "generated_token_count": 6,
                    "covered_records": 3,
                    "total_records": 3,
                    "coverage_ratio": 1.0,
                },
                "limited": True,
            }
        ),
        encoding="utf-8",
    )

    marker = tmp_path / "preflight.ok"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_no_steering_smoke.py"),
            "--result-root",
            str(result_root),
            "--model-id",
            MODEL_ID,
            "--marker",
            str(marker),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not marker.exists()


def test_smoke_uses_result_summary_and_checkpoint_manifest_responses(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results" / "Qwen--Qwen3-4B"
    checkpoint_root = tmp_path / "checkpoints" / "Qwen--Qwen3-4B"
    result_root.mkdir(parents=True)
    bogus_result_response = result_root / "math500" / "responses.jsonl"
    bogus_result_response.parent.mkdir()
    bogus_result_response.write_text(
        json.dumps({"status": "error", "id": "wrong-root"}) + "\n",
        encoding="utf-8",
    )
    (result_root / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model_id": MODEL_ID,
                    "revision": model_spec(MODEL_ID).revision,
                },
                "ppl": {
                    "ppl": 1.5,
                    "selected_token_count": 6,
                    "generated_token_count": 6,
                    "covered_records": 3,
                    "total_records": 3,
                    "coverage_ratio": 1.0,
                },
                "limited": True,
                "scores": {"status": "pending"},
            }
        ),
        encoding="utf-8",
    )
    benchmark_ids: dict[str, list[str]] = {}
    for benchmark in ("harmbench", "mmlu_pro", "math500"):
        identifier = f"{benchmark}-0"
        benchmark_ids[benchmark] = [identifier]
        path = checkpoint_root / benchmark / "responses.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "id": identifier,
                    "benchmark": benchmark,
                    "model_id": MODEL_ID,
                    "status": "ok",
                    "generated_token_count": 2,
                    "selected_generated_token_logprobs": [-0.1, -0.2],
                    "metadata": {
                        "provenance": {
                            "model_id": MODEL_ID,
                            "revision": model_spec(MODEL_ID).revision,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    config = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_slug": model_spec(MODEL_ID).slug,
        "model_revision": model_spec(MODEL_ID).revision,
        "limited": True,
        "benchmark_ids": benchmark_ids,
        "benchmark_manifest": {
            benchmark: {
                "loaded_count": 1,
                "expected_count": 1,
                "complete": False,
                "limited": True,
            }
            for benchmark in benchmark_ids
        },
    }
    (checkpoint_root / "manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": hashlib.sha256(
                    json.dumps(
                        config,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "config": config,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_no_steering_smoke.py"),
            "--result-root",
            str(result_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--model-id",
            MODEL_ID,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
