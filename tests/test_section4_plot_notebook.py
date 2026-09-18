"""The editing notebook must be a standalone plotting entrypoint."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / 'notebooks/section4_attention_bounds.ipynb'


def test_notebook_draws_without_importing_plot_script(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    assert 'scripts/plot_section4_long.py' not in source
    assert 'plotter.' not in source
    assert 'def combined_figure(' in source
    assert "set_yscale('log')" in source

    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/section4_long_summary.json').write_text(
        (ROOT / 'results/section4_long_summary.json').read_text())
    (tmp_path / 'figs').mkdir()
    (tmp_path / 'logs').mkdir()
    notebook.cells.append(nbformat.v4.new_code_cell("""
fig = combined_figure(-1)
left, right = fig.axes
for ax, label in ((left, '(a)'), (right, '(b)')):
    assert ax.get_title() == ''
    captions = [item for item in ax.texts if item.get_text() == label]
    assert len(captions) == 1
    assert captions[0].get_ha() == 'center'
    assert captions[0].get_position()[0] == .5
    assert captions[0].get_position()[1] < 0
assert right.get_ylim()[0] == 0
assert all((path.vertices[:, 1] >= 0).all()
           for band in right.collections for path in band.get_paths())
expected = [row['error']['mean'] for row in stats
            if row['layer'] == -1 and row['figure'] == 'prompt']
np.testing.assert_allclose(right.lines[0].get_ydata(), expected)
plt.close(fig)
"""))
    NotebookClient(notebook, timeout=120, kernel_name='python3',
                   resources={'metadata': {'path': str(tmp_path)}}).execute()

    outputs = [output for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.get('outputs', [])]
    assert sum('image/svg+xml' in output.get('data', {}) for output in outputs) == 6
    assert not any(output.output_type == 'error' for output in outputs)
    assert not any('image/png' in output.get('data', {}) for output in outputs)
    assert len(list((tmp_path / 'figs').glob('section4_*_combined.pdf'))) == 6
    assert all(path.read_bytes().startswith(b'%PDF')
               for path in (tmp_path / 'figs').glob('*.pdf'))
