import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('native_plot', Path(__file__).parents[1] / 'scripts/plot_section4_native.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_query_and_head_averages_precede_between_example_statistics():
    rows = []
    for index, head, errors in [(1,0,[1.,3.,99.]),(1,16,[5.,7.,99.]),(2,0,[9.,11.,99.]),(2,16,[13.,15.,99.])]:
        rows.append(dict(identity=dict(index=index,head=head,condition={'id':'m4_b1_g0','m':4,'b':1,'g':0}),
                         queries=[10,11,12], errors=errors,query_sets={'common':[10,11], 'exposed':[10,11,12], 'all':[10,11,12]},
                         diagnostic_dimension=120,fit={'final_error':1e-8},expanded=False))
    stats = module.token_statistics(rows)
    assert len(stats) == 1
    assert stats[0]['common']['n'] == 2
    assert stats[0]['common']['mean'] == pytest.approx(8.)
    assert stats[0]['common']['std'] == pytest.approx(32**.5)


def test_incomplete_head_coverage_is_rejected():
    rows = [dict(identity={'index':1,'head':0,'condition':{'id':'m4_b1_g0'}})]
    with pytest.raises(ValueError, match='coverage'):
        module.check_coverage(rows, [1], [0,16], [{'id':'m4_b1_g0'}])


def test_full_coverage_cannot_contain_duplicate_records():
    rows = [dict(identity={'index':1,'head':0,'condition':{'id':'m4_b1_g0'}})] * 2
    with pytest.raises(ValueError, match='coverage'):
        module.check_coverage(rows, [1], [0], [{'id':'m4_b1_g0'}])


def test_eos_audit_counts_behaviors_once_and_marks_post_eos_queries():
    rows = [dict(identity={'index':i},first_eos=eos) for i,eos in [(1,50),(1,50),(2,200),(2,200)]]
    audit = module.trace_statistics(rows)
    assert audit['examples'] == 2
    assert audit['eos_at_or_before_first_common_query'] == 1
    assert audit['eos_at_or_before_reference'] == 2
    assert audit['common_positions_at_or_after_eos'] == 184
    assert audit['common_positions_total'] == 256
    with pytest.raises(ValueError, match='inconsistent EOS'):
        module.trace_statistics(rows + [dict(identity={'index':1},first_eos=99)])


def test_log_panel_remains_readable_when_standard_deviation_crosses_zero(monkeypatch):
    stats = [{**c, 'common':{'mean':.1,'std':.2}, 'dimension':{'mean':120.,'std':0.}}
             for c in module.conditions()]
    limits = []
    def capture(fig, name):
        if name == 'section4_native_token_error':
            limits.append(fig.axes[0].get_ylim())
        module.plt.close(fig)
    monkeypatch.setattr(module, 'save', capture)
    controls = [dict(control=name,count=count,dimension={'mean':120.,'std':0.})
                for name in ['redundant','independent'] for count in module.LENGTHS]
    module.figures(stats, [], controls, 100)
    lower, upper = limits[0]
    assert .001 < lower < .1
    assert upper >= .3
