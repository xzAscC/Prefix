"""Command-line orchestration; CPU-only listing never initializes models/judges."""
from __future__ import annotations

import argparse
import contextlib
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .data import load_mmlu_pro
from .paper_backend import PaperBackend
from .paper_data import prepare_task
from .paper_experiments import PaperExperiment
from .paper_protocol import Checkpoint, PaperJudge
from .judge import GeminiJudge
from .runner import load_config, mmlu_prompt, tee_stdout


def jobs(config, model, task, study):
    models = list(config['models']) if model == 'all' else [model]
    tasks = config['tasks'] if task == 'all' else [task]
    if any(item not in config['models'] for item in models) or any(item not in config['tasks'] for item in tasks):
        raise ValueError('unknown model or task')
    if study == 'duration':
        if 'olmo3-7b' not in models or 'safety' not in tasks:
            raise ValueError('duration study is defined for olmo3-7b / safety')
        return [('olmo3-7b', 'safety')]
    return [(model_name, task_name) for model_name in models for task_name in tasks]


def candidate_layers(depth, fractions):
    if depth < 3 or not fractions or any(not 0 < value < 1 for value in fractions):
        raise ValueError('need depth >= 3 and nonempty interior layer fractions')
    return sorted({max(1, min(depth - 2, round((depth - 1) * value))) for value in fractions})


def implementation_digest():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted([*root.glob('paper_*.py'), root/'data.py', root/'judge.py', root/'runner.py', root/'steering.py']):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/paper.yaml'))
    parser.add_argument('--model', default='all')
    parser.add_argument('--task', default='all')
    parser.add_argument('--study', choices=['models', 'duration'], default='models')
    parser.add_argument('--run-id', default='paper')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--layer', type=int, help='override candidate layer set; zero-based')
    parser.add_argument('--limit', type=int, help='smoke run: truncate every split and direction class')
    parser.add_argument('--list', action='store_true', help='list jobs without downloading or running anything')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    matrix = jobs(config, args.model, args.task, args.study)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.list:
        for model_name, task in matrix:
            print(f'{args.study}: {model_name} / {task}')
        return
    # Bind checkpoints to the full protocol and implementation. Changed model,
    # prompts, datasets, smoke limits, or code require a new run ID.
    protocol = config['protocol']
    digest = implementation_digest()
    backend = None
    loaded_model = None
    for model_name, task in matrix:
        name = f'{args.run_id}.{args.study}.{model_name}.{task}'
        metadata = {'model_name': model_name, 'model': config['models'][model_name],
                    'study': args.study, 'task': task, 'config': config,
                    'implementation_sha256': digest, 'smoke_limit': args.limit,
                    'layer_override': args.layer, 'device': args.device}
        store = Checkpoint(args.root/'checkpoints', name, metadata)
        with tee_stdout(args.root/'logs'/f'{name}.log'), contextlib.redirect_stderr(sys.stdout):
            print(f'Starting {name}; smoke_limit={args.limit}', flush=True)
            def prepare():
                dataset = prepare_task(task, n_direction=protocol['n_direction'],
                                       n_tune=protocol['n_tune'], seed=protocol['seed'])
                mmlu = {}
                if task not in {'boxed', 'plain'}:
                    for split, source in [('tune', 'validation'), ('test', 'test')]:
                        mmlu[split] = [dict(id=f'{source}-{i}', prompt=mmlu_prompt(row['question'], row['options']),
                                           answer_letter=row['answer_letter']) for i, row in enumerate(load_mmlu_pro(source))]
                if args.limit is not None:
                    dataset = {key: values[:args.limit] for key, values in dataset.items()}
                    mmlu = {key: values[:args.limit] for key, values in mmlu.items()}
                return {'data': dataset, 'mmlu': mmlu}
            prepared = store.compute('datasets', prepare)
            if loaded_model != model_name:
                if backend is not None:
                    del backend
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                model = config['models'][model_name]
                backend = PaperBackend.load(model['id'], model['revision'], args.device)
                loaded_model = model_name
            layers = candidate_layers(len(backend.layers), protocol['layer_fractions'])
            if args.study == 'duration':
                layers = [protocol['duration_layer']]
            if args.layer is not None:
                layers = [args.layer]
            if any(layer < 0 or layer >= len(backend.layers) for layer in layers):
                raise ValueError('layer is outside the selected model')
            gemini = GeminiJudge(model=config['judge']['model'], region=config['judge']['region'])
            judge = SimpleNamespace(control=PaperJudge(gemini._call).control, judge_math=gemini.judge_math)
            experiment = PaperExperiment(protocol, task, backend, judge, store,
                                         prepared['data'], prepared['mmlu'], layers=layers,
                                         alternatives=config['models'][model_name].get('alternative_policies', False),
                                         study=args.study, metadata=metadata,
                                         summary_path=args.root/'results'/f'{name}.json')
            experiment.run()
            print(f'Completed {name}', flush=True)
            del experiment
