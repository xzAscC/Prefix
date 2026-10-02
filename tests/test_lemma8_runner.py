import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('lemma8',Path(__file__).parents[1]/'scripts/run_lemma8.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_sweeps_have_unique_conditions_and_include_full_and_empty_increment():
    rows=m.conditions()
    assert len(rows)==len({r['id'] for r in rows})
    assert any(r['g']==0 and r['schedule']=='generated' for r in rows)
    assert {r['alpha'] for r in rows} == {0,.25,.5,1,2}
    assert {r['m'] for r in rows} == {1,8,32,128}
    assert {r['k'] for r in rows} == {-1,1,4,16,64}


def test_supports_are_nested_and_exact_for_prediction_128():
    short,long=m.supports(10,dict(k=1,g=127,schedule='generated'))
    assert short==[9]
    assert long==[9]+list(range(10,137))
    assert m.supports(10,dict(k=16,g=1,schedule='generated')) is None
    short,long=m.supports(10,dict(k=-1,g=127,schedule='full'))
    assert short==list(range(10))
    assert long==list(range(137))
