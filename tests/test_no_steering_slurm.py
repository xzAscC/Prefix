from __future__ import annotations

import re
import os
import subprocess
from pathlib import Path

SCRIPTS = sorted(Path(__file__).parents[1].glob("scripts/sdsc*.sbatch"))
ROOT = Path(__file__).parents[1]


def test_sdsc_gpu_scripts_exist() -> None:
    assert SCRIPTS, "missing SDSC GPU Slurm scripts"


def test_sdsc_gpu_scripts_request_one_gpu_without_gres() -> None:
    assert SCRIPTS, "missing SDSC GPU Slurm scripts"
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "--gpus=1" in text, script
        assert "--gres" not in text, script


def test_sdsc_gpu_scripts_do_not_download_models() -> None:
    assert SCRIPTS, "missing SDSC GPU Slurm scripts"
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert not re.search(
            r"(?:^|\s)(?:python\s+-m\s+)?(?:huggingface-cli|hf|wget|curl|git\s+clone)\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ), script
        assert "from_pretrained" not in text, script


def test_preflight_runs_bounded_generation_for_every_model_before_marker() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    model_ids = (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-14B",
        "allenai/Olmo-3-7B-Think",
        "allenai/Olmo-3-32B-Think",
    )
    assert "--phase generate" in script
    assert "--limit 1" in script
    assert "--max-tokens" in script
    assert "--max-model-len" in script
    assert "--gpu-memory-utilization" in script
    assert "summary.json" in script
    assert script.count('"$UV_BIN" run python scripts/run_no_steering.py') == 1
    assert 'for model_id in "${model_ids[@]}"' in script
    for model_id in model_ids:
        assert model_id in script
    assert script.index("--phase generate") < script.index("mv -f")
    assert "--phase check" in script


def test_preflight_verification_requires_all_smoke_outputs() -> None:
    script = (ROOT / "scripts" / "sdsc_verify_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert "summary.json" in script
    for model_id in (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-14B",
        "allenai/Olmo-3-7B-Think",
        "allenai/Olmo-3-32B-Think",
    ):
        assert model_id in script
    assert '"$PREFLIGHT_OUTPUT_ROOT/$model_slug/summary.json"' in script
    assert "Qwen/Qwen3-4B/summary.json" not in script
    assert "validate_no_steering_formal.py" in script


def test_launcher_validates_assets_before_any_submission() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert script.index("no_steering_preflight.py") < script.index("sbatch")
    assert "--phase check" in script
    assert "--manifest" in script and "--marker" in script
    assert "--gpus" not in script


def test_launcher_resolves_and_exports_absolute_uv_before_submission() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    uv_resolution = 'UV_BIN="$(command -v uv)"'
    assert uv_resolution in script
    assert '[[ "$UV_BIN" == /* && -x "$UV_BIN" ]]' in script
    assert script.index(uv_resolution) < script.index("sbatch")
    assert "UV_BIN=$UV_BIN" in script
    assert "--export=ALL" not in script


def test_launcher_missing_uv_fails_before_any_sbatch_submission(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    sbatch_called = tmp_path / "sbatch-called"
    (commands / "sbatch").write_text(
        f'#!/usr/bin/env bash\nprintf called > "{sbatch_called}"\nexit 0\n',
        encoding="utf-8",
    )
    _ = (commands / "sbatch").chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{commands}:/usr/bin:/bin"
    result = subprocess.run(
        [str(ROOT / "scripts" / "sdsc_launch_no_steering.sh"), str(workspace)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not sbatch_called.exists()


def test_sbatch_scripts_execute_explicit_uv_bin() -> None:
    for name in ("sdsc_no_steering_preflight.sbatch", "sdsc_no_steering_job.sbatch"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert ': "${UV_BIN:?UV_BIN must be supplied explicitly}"' in script
        assert '"$UV_BIN" run python' in script
        assert re.search(r"(?m)^\s*uv run python", script) is None


def test_launcher_validates_all_dataset_caches_before_any_submission() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    cache_check = script.index("--dataset-cache-root")
    first_submission = script.index("sbatch")
    assert cache_check < first_submission
    assert "HF_HOME" in script[cache_check:first_submission]


def test_launcher_isolates_each_submission_from_stale_preflight_state() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert 'rm -f "$PREFLIGHT_MARKER"' in script
    assert "RUN_ID=" in script
    assert "RUN_ROOT=" in script
    assert 'mkdir -p "$WORKSPACE/logs"' in script
    assert "RUN_ROOT=$RUN_ROOT" in script
    assert "PREFLIGHT_MARKER=$PREFLIGHT_MARKER" in script


def test_formal_jobs_validate_read_only_and_use_canonical_run_roots() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_job.sbatch").read_text(
        encoding="utf-8"
    )
    assert '--model-id "$MODEL_ID"' in script
    assert "--manifest" not in script
    assert "--marker" not in script
    assert "$RUN_ROOT/checkpoints" in script
    assert "$RUN_ROOT/results" in script
    assert '"$MODEL_SLUG"' in script
    assert "--phase score" not in script


def test_formal_job_passes_hf_cache_to_generation() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_job.sbatch").read_text(
        encoding="utf-8"
    )
    assert '--cache-root "$HF_HOME"' in script


def test_formal_jobs_retain_exact_sds_c_constraints() -> None:
    for name in ("sdsc_no_steering_preflight.sbatch", "sdsc_no_steering_job.sbatch"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "#SBATCH --account=ohs122" in text
        assert "#SBATCH --partition=nairr-gpu-shared" in text
        assert "#SBATCH --qos=nairr-gpu-shared-normal" in text
        assert "#SBATCH --gpus=1" in text
        assert "--gres" not in text


def test_local_handoff_scores_staged_responses_without_gpu_or_slurm() -> None:
    script = (ROOT / "scripts" / "score_no_steering_local.sh").read_text(
        encoding="utf-8"
    )
    assert "--phase score" in script
    assert "gemini-3.5-flash-lite" in script
    assert "sbatch" not in script
    assert "--gpus" not in script
    assert '--checkpoint-root "$RESPONSE_ROOT"' in script
    assert '--output-root "$OUTPUT_ROOT"' in script


def test_preflight_smoke_failure_is_checked_before_atomic_marker_write() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    assert "validate_no_steering_smoke.py" in script
    assert script.index("validate_no_steering_smoke.py") < script.index("mv -f")
    assert "summary.json" not in script or "summary" in script


def test_launcher_exports_only_explicit_non_secret_inputs() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert '--export="ALL' not in script
    assert "--export=ALL" not in script
    assert "WORKSPACE=$WORKSPACE" in script
    assert "HF_HOME=$HF_HOME" in script
    assert "RUN_ROOT=$RUN_ROOT" in script


def test_slurm_scripts_do_not_release_jobs_from_exit_traps() -> None:
    for name in ("sdsc_no_steering_preflight.sbatch", "sdsc_no_steering_job.sbatch"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "scontrol release" not in script


def test_preflight_marker_uses_unique_temp_file_and_cleans_it_up() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    assert "mktemp" in script
    assert "PREFLIGHT_MARKER}.tmp.${SLURM_JOB_ID" not in script
    assert 'rm -f -- "$tmp_marker"' in script


def test_preflight_uses_one_hour_walltime_and_canonical_aggregate_root() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --time=01:00:00" in script
    assert '--output-root "$PREFLIGHT_OUTPUT_ROOT"' in script
    assert '--output-root "$PREFLIGHT_OUTPUT_ROOT/$model_slug"' not in script


def test_preflight_generation_uses_aggregate_checkpoint_root() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    assert '--checkpoint-root "$PREFLIGHT_CHECKPOINT_ROOT"' in script
    generation = script.split("validate_no_steering_smoke.py", 1)[0]
    assert (
        '--checkpoint-root "$PREFLIGHT_CHECKPOINT_ROOT/$model_slug"' not in generation
    )


def test_preflight_smoke_validator_receives_model_checkpoint_root() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    assert '--result-root "$PREFLIGHT_OUTPUT_ROOT/$model_slug"' in script
    assert script.index(
        '--result-root "$PREFLIGHT_OUTPUT_ROOT/$model_slug"'
    ) < script.index('--checkpoint-root "$PREFLIGHT_CHECKPOINT_ROOT/$model_slug"')


def test_formal_job_passes_aggregate_roots_to_generation() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_job.sbatch").read_text(
        encoding="utf-8"
    )
    assert '--checkpoint-root "$CHECKPOINT_DIR"' in script
    assert '--output-root "$FORMAL_OUTPUT_ROOT"' in script
    assert '--checkpoint-root "$CHECKPOINT_DIR/$MODEL_SLUG"' not in script
    assert '--output-root "$FORMAL_OUTPUT_ROOT/$MODEL_SLUG"' not in script


def test_verifier_passes_model_scoped_roots_to_formal_validator() -> None:
    script = (ROOT / "scripts" / "sdsc_verify_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert '--result-root "$RUN_ROOT/results/$model_slug"' in script
    assert '--checkpoint-root "$RUN_ROOT/checkpoints/$model_slug"' in script


def test_launcher_chains_formal_jobs_after_successful_preflight_only() -> None:
    script = (ROOT / "scripts" / "sdsc_launch_no_steering.sh").read_text(
        encoding="utf-8"
    )
    assert '"--dependency=afterok:${preflight_job_id}"' in script


def test_formal_child_does_not_receive_shared_asset_manifest_writers() -> None:
    script = (ROOT / "scripts" / "sdsc_no_steering_job.sbatch").read_text(
        encoding="utf-8"
    )
    assert "--manifest" not in script
    assert "--marker" not in script
