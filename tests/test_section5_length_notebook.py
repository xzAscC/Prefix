"""The standalone figure uses one matched cohort for steering and prompt."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).parents[1]


def test_section5_length_notebook(tmp_path):
    notebook = nbformat.read(ROOT / 'notebooks/section5_input_length.ipynb', as_version=4)
    (tmp_path / 'results').mkdir()
    (tmp_path / 'figs').mkdir()
    (tmp_path / 'results/section5_fixed_summary.json').write_bytes(
        (ROOT / 'results/section5_fixed_summary.json').read_bytes())
    for path in (ROOT / 'results').glob('section5_fixed_[0-9][0-9][0-9].json'):
        state = __import__('json').loads(path.read_text())
        if state['units']['17_0']['base_length'] >= 64:
            (tmp_path / 'results' / path.name).write_bytes(path.read_bytes())
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert len(cohort) == 100
assert len(fig.axes) == 2
assert all(len(ax.lines) == 2 for ax in fig.axes)
assert all(ax.lines[1].get_label() == 'Prompt (m=8)' for ax in fig.axes)
assert all(ax.lines[0].get_label() == 'Steering' for ax in fig.axes)
assert all(len(ax.lines[0].get_xdata()) == 5 for ax in fig.axes)
assert all(ax.get_xticklabels()[-1].get_text() == 'Full' for ax in fig.axes)
assert all(ax.lines[1].get_ydata()[0] == baseline[field]['mean']
           for ax, field in zip(fig.axes, ('R', 'C')))
assert (ROOT / 'figs/section5_input_length.pdf').read_bytes().startswith(b'%PDF')
"""))
    NotebookClient(notebook, timeout=120, kernel_name='python3',
                   resources={'metadata': {'path': str(tmp_path)}}).execute()
