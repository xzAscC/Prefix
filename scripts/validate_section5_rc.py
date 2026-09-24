"""Audit complete Section 5 condition coverage and saved generation evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_row(row):
    if not row['eligible']:
        require(bool(row.get('reason')), 'Missing ineligibility reason')
        return
    require(math.isfinite(row['R']) and -1-1e-10 <= row['R'] <= 1+1e-10, 'Invalid R')
    require(math.isfinite(row['output_norm']) and row['output_norm'] > 0, 'Invalid output norm')
    if row['method'] == 'unsteered' or (row['method'] != 'prompt' and row['alpha'] == 0):
        require(abs(row['R']-1) < 1e-10, 'Zero strength must preserve baseline')
    if row['C'] is None:
        require(row.get('concept_defined') is False, 'Undefined C must be explicit')
        require(all(row[k] is None for k in ['C0','delta_C','delta_perp']), 'Partial undefined concept record')
        return
    require(all(math.isfinite(row[k]) for k in ['C','C0','delta_C','delta_perp']), 'Nonfinite concept metric')
    require(abs(row['C']) <= 1+1e-10 and abs(row['C0']) <= 1+1e-10, 'Invalid cosine')
    require(abs(row['delta_C'] - (row['C']-row['C0'])) < 1e-10, 'Invalid delta C')
    rhs = 1-(row['delta_C']**2 + row['delta_perp'])/2
    require(abs(row['R']-rhs) < 1e-10, 'Section 5 identity failed')


def check_eos(row, tokens, eos_ids, repair):
    stop = next((i+1 for i, token in enumerate(tokens) if token in eos_ids), None)
    continued = stop is not None and stop < len(tokens)
    if not repair:
        require(row['first_eos'] == stop and row['continued_after_eos'] == continued,
                'EOS metadata disagrees with model-configured stop tokens')
    return {**row, 'first_eos': stop, 'continued_after_eos': continued, 'eos_token_ids': eos_ids}


def audit(study, limit, repair_eos=False):
    checked, undefined, ineligible = 0, 0, 0
    shared_manifest = None
    eos_ids = None
    repaired_records = 0
    for index in range(limit):
        path = ROOT / f'results/section5_{study}_{index:03d}.json'
        state = json.loads(path.read_text())
        manifest = state['manifest']
        changed = False
        if study == 'generation' and eos_ids is None:
            from transformers import GenerationConfig
            config = GenerationConfig.from_pretrained(manifest['model'], revision=manifest['revision'], local_files_only=True)
            eos_ids = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
        require(manifest['index'] == index, f'Misassigned result: {path}')
        shared = {k:v for k,v in manifest.items() if k not in ['index','activation_sha256']}
        require(shared_manifest is None or shared_manifest == shared, 'Mixed experiment manifests')
        shared_manifest = shared
        conditions = {c['id'] for c in manifest['conditions']}
        if study == 'fixed':
            expected = {f'{layer}_{head}' for layer in manifest['layers'] for head in manifest['heads']}
            require(set(state['units']) == expected, f'Incomplete heads: {path}')
            groups = []
            for unit in state['units'].values():
                require(unit['index'] == index, 'Misassigned head result')
                require({r['id'] for r in unit['rows']} == conditions, 'Incomplete fixed conditions')
                require(len(unit['rows']) == len(conditions), 'Duplicated fixed conditions')
                for comparison in unit['comparisons']:
                    if comparison.get('concept_defined') is False:
                        continue
                    require(abs(comparison['R_gap']-comparison['identity_rhs']) < 1e-10, 'Comparison identity failed')
                    require(comparison['R_gap'] >= comparison['lower_bound']-1e-10, 'Theorem inequality failed')
                groups.extend(unit['rows'])
        else:
            require(set(state['units']) == conditions, f'Incomplete generation conditions: {path}')
            groups = list(state['units'].values())
        for row in groups:
            validate_row(row)
            checked += 1
            ineligible += not row['eligible']
            if not row['eligible']:
                continue
            undefined += row['C'] is None
            if study == 'generation':
                checkpoint = ROOT / f'checkpoints/section5_generation_{index:03d}_{row["id"]}.json'
                saved = json.loads(checkpoint.read_text())
                require(saved['manifest'] == {**manifest, 'condition': next(c for c in manifest['conditions'] if c['id']==row['id'])}, 'Decode manifest mismatch')
                require(len(saved['tokens']) == row['generated_count'] == 128, 'Not exactly 128 generated tokens')
                require(all(isinstance(t,int) and t >= 0 for t in saved['tokens']), 'Invalid token IDs')
                fingerprint = hashlib.sha256(json.dumps(saved['tokens']).encode()).hexdigest()
                require(fingerprint == row['tokens_sha256'], 'Token fingerprint mismatch')
                require(len(saved['final_output']) == 128 and all(math.isfinite(v) for v in saved['final_output']), 'Invalid captured native head output')
                corrected = check_eos(row, saved['tokens'], eos_ids, repair_eos)
                if repair_eos and corrected != row:
                    state['units'][row['id']] = corrected
                    changed = True
                    repaired_records += 1
        if changed:
            write_json_atomic(path, state)
        print(f'{study} {index+1}/{limit}: audited', flush=True)
    return dict(study=study, examples=limit, records=checked, undefined_concept=undefined,
                ineligible=ineligible, complete=True, eos_metadata_repaired=repaired_records, manifest=shared_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['fixed','generation','both'], default='both')
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--repair-eos', action='store_true', help='Recompute EOS metadata from saved token evidence on complete examples only')
    args = parser.parse_args()
    with tee_stdout(ROOT / 'logs/section5_validation.log'):
        studies = ['fixed','generation'] if args.study=='both' else [args.study]
        for study in studies:
            result = audit(study, args.limit, args.repair_eos)
            write_json_atomic(ROOT / f'results/section5_{study}_audit.json', result)
            print(f'{study}: {result["records"]} records verified', flush=True)


if __name__ == '__main__':
    main()
