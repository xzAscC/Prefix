import hashlib
import json
import pytest
from prefix.trace_metadata import summarize_traces


def batch(inputs,prompt,tokens):
    return {'indices':list(range(len(inputs))), 'tokens':tokens,
            'prefix_hash':hashlib.sha256(json.dumps([x['input_ids']+prompt for x in inputs]).encode()).hexdigest()}


def test_reference_eos_distinguishes_before_and_at_target():
    inputs=[dict(index=i,input_ids=[i+1]) for i in range(3)]
    b=batch(inputs,[4],[[9,2,3],[1,2,8],[1,2,3]])
    result=summarize_traces(inputs,[4],[b],[8,9],3)
    assert result['examples']['0']['past_eos_at_prediction'] is True
    assert result['examples']['1']['past_eos_at_prediction'] is False
    assert result['past_eos_examples']==1
    assert result['total_examples']==3


def test_reference_trace_checks_prompt_hash_and_unique_input_ids():
    inputs=[dict(index=0,input_ids=[1])]
    b=batch(inputs,[4],[[9,2,3]])
    with pytest.raises(ValueError,match='prefix'):
        summarize_traces(inputs,[5],[b],[9],3)
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_traces(inputs,[4],[b,b],[9],3)
