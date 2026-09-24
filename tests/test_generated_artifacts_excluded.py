from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_figures_and_notebooks_are_not_committed():
    for name in ("figs", "notebooks"):
        directory = ROOT / name
        assert {path.name for path in directory.iterdir()} == {".gitkeep"}
