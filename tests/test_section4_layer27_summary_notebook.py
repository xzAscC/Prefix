import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/section4_layer27_summary.ipynb'


@pytest.mark.parametrize('directory', [ROOT, ROOT / 'notebooks'])
def test_summary_notebook_preserves_data_and_bands(directory, monkeypatch):
    monkeypatch.chdir(directory)
    notebook = json.loads(NOTEBOOK.read_text())
    scope = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'export' not in cell['metadata'].get('tags', []):
            exec(''.join(cell['source']), scope)
    fig = scope['make_figure']()
    a, b, c = fig.axes
    assert [len(ax.lines) for ax in fig.axes] == [2, 2, 3]
    assert a.get_xscale() == a.get_yscale() == 'linear'
    assert b.get_yscale() == c.get_yscale() == 'log'
    assert len(b.collections) == 2
    assert scope['DRIFT_CONDITION'] == 'm4_b1_g3'
    assert len(c.collections) == 3
    assert scope['diversity']['formal'] and scope['diversity']['examples']==100
    assert scope['diversity']['distinct_prompt_prefixes']['128']==100
    expected = sorted([r for r in scope['diversity']['token_statistics'] if r['m']==4 and r['b']==1],
                      key=lambda r:r['b']+r['g'])
    np.testing.assert_allclose(a.lines[0].get_ydata(), [r['dimension']['mean'] for r in expected])
    drift = [r for r in scope['native']['drift_statistics'] if r['condition']=='m4_b1_g3']
    for line, direction in zip(b.lines, ['sensitive', 'random']):
        series = sorted([r for r in drift if r['direction']==direction], key=lambda r:r['rho']['mean']
                        if r['rho']['mean']>=1e-9 else 0)
        np.testing.assert_allclose(line.get_ydata(), [r['native_error']['mean'] for r in series])
    old = [r for r in scope['legacy']['comparison'] if r['layer']==26 and r['figure']=='prompt']
    np.testing.assert_allclose(c.lines[-1].get_ydata(),
                               [r['error']['mean'] for r in sorted(old, key=lambda r:r['m'])])
    for ax, label in zip(fig.axes, ['(a)', '(b)', '(c)']):
        assert label in [t.get_text() for t in ax.texts]
        assert not ax.get_title()
        assert all('bound' not in line.get_label().lower() and line.get_label()!='null' for line in ax.lines)
    plt.close(fig)
