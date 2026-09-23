import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prompt_diversity', ROOT/'scripts/run_section4_prompt_diversity.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def test_manifest_uses_distinct_dataset_prompts_and_records_padding():
    cohort = dict(fingerprint='base', config={'prompt_slots':128}, examples=[
        dict(index=i,base_ids=[i]*130,input_tokens=130) for i in range(2)])
    records = [dict(BehaviorID=str(i),Behavior=f'Request {i}',ContextString='') for i in range(2)]
    manifest = module.build_manifest(cohort, records, Tokenizer())
    rows = manifest['examples']
    assert [r['prompt_source_index'] for r in rows] == [1,0]
    assert all(r['input_index'] != r['prompt_source_index'] for r in rows)
    assert all(len(r['prompt_ids'])==128 and r['added_text'] for r in rows)
    assert len({tuple(r['prompt_ids']) for r in rows})==2
    assert rows[0]['prompt_text'].startswith('Request 1')
    assert module.build_manifest(cohort, records, Tokenizer()) == manifest


def test_dimension_depends_on_rank_not_token_count():
    repeated = torch.tensor([[1.,0.,0.],[2.,0.,0.]],dtype=torch.float64)
    independent = torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    assert module.rank_diagnostics(repeated,1e-7)['dimension']==2
    assert module.rank_diagnostics(independent,1e-7)['dimension']==1


def test_resume_checks_identity_and_retains_finished_result(tmp_path):
    p=tmp_path/'result.json'; payload={'identity':{'example':1},'complete':True,'dimension':3}
    p.write_text(json.dumps(payload)); before=p.read_bytes()
    assert module.completed_result(p,{'example':1})==payload
    assert p.read_bytes()==before
    with pytest.raises(ValueError,match='identity'):
        module.completed_result(p,{'example':2})


def test_summary_averages_heads_before_sample_std():
    rows=[dict(identity=dict(input_index=i,head=h,condition={'id':'m1_b1_g0','m':1,'b':1,'g':0}),
               dimension=d) for i,h,d in [(0,0,2),(0,16,4),(1,0,6),(1,16,8)]]
    summary=module.summarize(rows)
    assert summary[0]['dimension']['n']==2
    assert summary[0]['dimension']['mean']==5
    assert summary[0]['dimension']['std']==pytest.approx(8**.5)
