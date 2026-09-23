"""Check the toy plot preserves its data and labels its illustrative status."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def test_toy_plot_is_standalone_and_preserves_values(tmp_path):
    notebook = nbformat.read(ROOT / 'notebooks/tradeoff_7b_toy.ipynb', as_version=4)
    for name in ('notebooks', 'figs'):
        (tmp_path / name).mkdir()
    notebook.cells.append(nbformat.v4.new_code_cell('''
assert len(fig.axes) == 1
assert len(DATA) == 14
assert next(row[2:] for row in DATA if row[:2] == ('All-token', 10)) == (22.042, 98.317)
assert next(row[2:] for row in DATA if row[:2] == ('Prefix-1', 10)) == (38.813, 83.127)
assert ax.get_xlim()[0] <= min(row[2] for row in DATA)
assert ax.get_xlim()[1] >= max(row[2] for row in DATA)
assert OUTPUT.name == 'steering_tradeoff_7b_toy.pdf'
'''))
    NotebookClient(notebook, timeout=90, kernel_name='python3',
                   resources={'metadata': {'path': str(tmp_path / 'notebooks')}}).execute()
    outputs = [o for c in notebook.cells if c.cell_type == 'code' for o in c.outputs]
    assert any('image/svg+xml' in o.get('data', {}) for o in outputs)
    assert not any('image/png' in o.get('data', {}) for o in outputs)
    assert (tmp_path / 'figs/steering_tradeoff_7b_toy.pdf').read_bytes().startswith(b'%PDF')
    assert not (tmp_path / 'figs/steering_tradeoff_7b.pdf').exists()
