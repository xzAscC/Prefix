import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('plot_lemma8',Path(__file__).parents[1]/'scripts/plot_lemma8.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_statistics_average_heads_within_behavior_before_averaging_behaviors():
    rows=[{'index':i,'error':e} for i,e in [(0,0.),(0,2.),(1,5.)]]
    result=m.mean_std(rows,'error')
    assert result['n']==2
    assert result['mean']==3.


def test_low_share_audit_requires_both_sides_and_excludes_zero_interventions():
    rows=[dict(index=i,alpha=a,extra_count=1,w_short=ws,w_long=wl,error=1e-4,value_shift_norm=1.,diameter_short=2.)
          for i,a,ws,wl in [(0,1,.001,.002),(1,1,.001,.9),(2,0,.001,.002)]]
    result=m.low_share_statistics(rows,.01)
    assert result['conditions']==1
    assert result['error']['n']==1
    assert result['violations']==0
