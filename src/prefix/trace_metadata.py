"""EOS provenance for the shared Section 4 continuation used in fixed replay."""
import hashlib
import json
from prefix.runner import write_json_atomic


def summarize_traces(inputs,prompt_ids,batches,eos_ids,target):
    examples={}
    for batch in batches:
        indices=batch['indices']
        prefixes=[inputs[i]['input_ids']+prompt_ids for i in indices]
        if hashlib.sha256(json.dumps(prefixes).encode()).hexdigest()!=batch['prefix_hash']:
            raise ValueError('Reference trace prefix hash mismatch')
        if len(indices)!=len(batch['tokens']):
            raise ValueError('Trace batch length mismatch')
        for i,tokens in zip(indices,batch['tokens']):
            if str(i) in examples:
                raise ValueError('Duplicate reference trace index')
            if inputs[i]['index']!=i or len(tokens)!=target:
                raise ValueError('Reference trace index or token count mismatch')
            stop=next((j+1 for j,token in enumerate(tokens) if token in eos_ids),None)
            examples[str(i)]=dict(first_eos=stop,past_eos_at_prediction=stop is not None and stop<target,
                                  tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
    return dict(total_examples=len(examples),target=target,eos_token_ids=eos_ids,
                complete=len(examples)==len(inputs),examples=examples,
                past_eos_examples=sum(x['past_eos_at_prediction'] for x in examples.values()))


def reference_trace_metadata(root):
    from transformers import GenerationConfig
    source=json.loads((root/'results/section4_long_manifest.json').read_text())
    inputs=json.loads((root/'data/section5_inputs.json').read_text())
    config=GenerationConfig.from_pretrained(source['model'],revision=source['revision'],local_files_only=True)
    eos=config.eos_token_id if isinstance(config.eos_token_id,list) else [config.eos_token_id]
    batches=[json.loads(p.read_text()) for p in sorted((root/'checkpoints').glob('section4_long_decode_*.json'))]
    result=summarize_traces(inputs,source['prompt_ids'][:source['generation_prompt_tokens']],batches,eos,128)
    if not result['complete']:
        raise ValueError('Shared reference traces are incomplete')
    result.update(model=source['model'],revision=source['revision'],generation_prompt_tokens=source['generation_prompt_tokens'])
    write_json_atomic(root/'results/section5_reference_trace_metadata.json',result)
    return result
