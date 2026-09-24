import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("launch_dir", [ROOT, ROOT / "notebooks"])
def test_alpha_plot_preserves_all_scores(launch_dir, monkeypatch):
    notebook = json.loads((ROOT / "notebooks/alpha.ipynb").read_text())
    monkeypatch.chdir(launch_dir)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert "TASK_AXES" in namespace, "The notebook needs a reproducible plotting cell"
    rows = namespace["DECAY_DATA"]
    assert len(rows) == 35
    assert len({(task, method) for task, method, _, _ in rows}) == 35
    for task, ax in namespace["TASK_AXES"].items():
        expected = sorted((x, y) for name, _, x, y in rows if name == task)
        actual = sorted((float(line.get_xdata()[0]), float(line.get_ydata()[0]))
                        for line in ax.lines)
        assert actual == expected
        for x, y in actual:
            assert ax.get_xlim()[0] < x < ax.get_xlim()[1]
            assert ax.get_ylim()[0] < y < ax.get_ylim()[1]
    assert namespace["OUTPUT_PDF"].read_bytes().startswith(b"%PDF")
