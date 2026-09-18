import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('plot_long', Path(__file__).parents[1] / 'scripts/plot_section4_long.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_separate_layers_fixed_cohort_and_prompt_level_std():
    rows = []
    for index, n in [(0,30),(1,128),(2,400)]:
        for layer in [2,8]:
            for head in [0,16]:
                for m,k,g in [(4,1,0),(4,128,0),(4,4,4),(1,1,0)]:
                    rows.append(dict(index=index,base_length=n,layer=layer,head=head,m=m,k=k,g=g,
                                     eligible=n>=k,error=10*layer+index+head/16,
                                     bound_certified=100+10*layer+index+head/16))
    stats = module.statistics(rows)
    for item in stats:
        assert item['error']['n'] == (3 if item['figure'] == 'prompt' else 2)
        expected_std = 1 if item['figure'] == 'prompt' else 2**-.5
        assert item['error']['std'] == pytest.approx(expected_std)
    item = next(r for r in stats if r['layer']==2 and r['figure']=='steering' and r['k']==1)
    assert item['error']['mean'] == 22
    assert {r['layer'] for r in stats} == {2,8}


def test_loader_rejects_missing_head_measurements(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(module,'ROOT',tmp_path)
    (tmp_path/'results').mkdir()
    state={'manifest':{'layers':[2], 'heads':[0,16], 'conditions':[{'id':'test'}],
                       'jacobian_anchor':'last_input_token'},
           'units':{'2_0/test':{'index':0}}}
    (tmp_path/'results/section4_long_000.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Incomplete'):
        module.load_rows(1)


def test_trends_report_decreases_without_forcing_monotonicity():
    stats = [dict(layer=2,figure='steering',m=4,k=k,g=0,
                  error={'mean':value,'std':0.,'n':2})
             for k,value in [(1,3.),(2,1.),(128,2.)]]
    result = module.trends(stats)[0]
    assert result['monotone_nondecreasing'] is False
    assert result['last_minus_first'] == -1.


def test_layer_mean_uses_paired_inputs_before_computing_std():
    rows = []
    for j,layer in enumerate([2,8,17,26,33]):
        for index in [0,1]:
            for head in [0,16]:
                error = 10*(j+1) + index*[1,-2,3,-4,5][j] + (-2 if head==0 else 2)
                rows.append(dict(layer=layer,index=index,head=head,base_length=128,
                                 m=4,k=1,g=0,eligible=True,error=error,bound_certified=error+100))
    stats = module.layer_mean_statistics(rows)
    assert len(stats)==2
    for row in stats:
        assert row['layer']==-1
        assert row['error']['n']==2
        assert row['error']['mean']==pytest.approx(30.3)
        assert row['error']['std']==pytest.approx(.6/2**.5)
        assert row['bound']['mean']==pytest.approx(130.3)


def test_combined_figure_has_shared_error_bound_axis_and_prompt_panel():
    stats = [dict(layer=2,figure=figure,m=m,k=k,g=g,
                  error=dict(mean=2.,std=.5,n=95 if figure=='steering' else 400),
                  bound=dict(mean=5.,std=1.,n=95 if figure=='steering' else 400))
             for figure,m,k,g in [('steering',4,1,0),('steering',4,4,0),
                                   ('steering',4,4,4),('prompt',1,1,0),('prompt',4,1,0)]]
    fig = module.combined_figure(stats,2)
    left,right = fig.axes
    assert len(fig.axes)==2
    measured = [line for line in left.lines if 'Measured' in line.get_label()]
    bounds = [line for line in left.lines if 'Bound' in line.get_label()]
    assert len(measured)==len(bounds)==2
    assert all(line.get_linestyle()=='-' for line in measured)
    assert all(line.get_linestyle()=='--' for line in bounds)
    assert right.lines[0].get_color() not in {line.get_color() for line in measured}
    assert all(y==2 for line in measured for y in line.get_ydata())
    assert all(y==5 for line in bounds for y in line.get_ydata())
    module.plt.close(fig)
