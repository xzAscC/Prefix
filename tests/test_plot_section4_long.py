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
