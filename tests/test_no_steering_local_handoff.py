from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_handoff_scores_staged_responses_without_gpu_or_scheduler():
    script = (ROOT / "scripts" / "score_no_steering_local.sh").read_text(encoding="utf-8")
    assert "--phase score" in script
    assert "gemini-3.5-flash-lite" in script
    assert "sbatch" not in script
    assert "--gpus" not in script
    assert '--checkpoint-root "$RESPONSE_ROOT"' in script
    assert '--output-root "$OUTPUT_ROOT"' in script
