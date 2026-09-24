from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_and_run_artifacts_are_not_committed():
    for name in ("figs", "notebooks", "data", "results"):
        directory = ROOT / name
        assert {path.name for path in directory.iterdir()} == {".gitkeep"}
