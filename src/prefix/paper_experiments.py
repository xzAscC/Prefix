"""Held-out tuning and evaluation for the paper's four-model/five-task study."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .paper_backend import Intervention
from .paper_data import BASELINE_PROMPTS
from .paper_operators import CoastOperator
from .paper_protocol import choose_setting, fit_probe
from .runner import parse_answer_letter, write_json_atomic
from .steering import dim_direction, mean_hidden_norm


def setting_id(setting: dict) -> str:
    digest = hashlib.sha256(json.dumps(setting, sort_keys=True).encode()).hexdigest()[:16]
    return f"{setting['method']}-{digest}"


class PaperExperiment:
    def __init__(self, config, task, backend, judge, checkpoint, data, mmlu, *, layers,
                 alternatives, summary_path, study='models', metadata=None):
        self.config, self.task, self.backend, self.judge = config, task, backend, judge
        self.store, self.data, self.mmlu = checkpoint, data, mmlu
        self.layers, self.alternatives = layers, alternatives
        self.summary_path = Path(summary_path)
        self.study = study
        if study not in {'models', 'duration'}:
            raise ValueError('unknown paper study')
        if study == 'duration' and (task != 'safety' or len(layers) != 1):
            raise ValueError('duration study requires safety and one fixed layer')
        self.activations = {}
        self.vectors = {}
        self.coast_operator = None
        self.probe = None
        self.report = {'task': task, 'study': study, 'metadata': metadata or {}, 'tune': {}, 'test': {}}

    def save_report(self):
        write_json_atomic(self.summary_path, self.report)

    def capture_directions(self):
        if self.activations:
            return
        thinking = self.config['thinking']['math' if self.task in {'boxed', 'plain'} else 'concept']
        for label in ('positive', 'negative'):
            rows = []
            for index, text in enumerate(self.data[label]):
                def compute(text=text):
                    values = self.backend.capture([{'role': 'user', 'content': text}], self.layers, thinking=thinking)
                    return {str(layer): value.tolist() for layer, value in values.items()}
                rows.append(self.store.compute(f'direction/{label}/{index}', compute))
            for layer in self.layers:
                self.activations[label, layer] = torch.tensor([row[str(layer)] for row in rows])
        for layer in self.layers:
            pos, neg = self.activations['positive', layer], self.activations['negative', layer]
            self.vectors[layer] = (dim_direction(pos, neg), mean_hidden_norm(torch.cat([pos, neg])))

    def intervention(self, setting):
        if setting['method'] in {'unsteered', 'prompting'}:
            return None
        self.capture_directions()
        layer = setting['layer']
        direction, mean_norm = self.vectors[layer]
        spec = Intervention(layer=layer, direction=direction, operator=setting['operator'],
                            policy=setting['policy'], coefficient=setting.get('strength', 1.) * mean_norm,
                            length=setting.get('length', 5), tau=self.config['exponential_tau'],
                            das_top_p=self.config['das_top_p'], das_maximum=self.config['das_maximum'],
                            act_amplitude=self.config['act_amplitude'], act_bias=self.config['act_bias'])
        if spec.operator == 'coast':
            if self.coast_operator is None:
                x = torch.cat([self.activations['positive', layer], self.activations['negative', layer]])
                second_moment = x.double().T @ x.double() / len(x)
                self.coast_operator = CoastOperator(direction.to(self.backend.device), second_moment.to(self.backend.device))
            spec.coast = self.coast_operator
            spec.target_cosine = math.cos(math.radians(setting['angle']))
        if spec.policy == 'act':
            if self.probe is None:
                def train():
                    weight, bias = fit_probe(self.activations['positive', layer], self.activations['negative', layer],
                                            self.config['probe_regularization'], self.config['probe_max_iter'])
                    return {'weight': weight.tolist(), 'bias': bias}
                self.probe = self.store.compute(f'probe/{layer}', train)
            spec.probe_weight = torch.tensor(self.probe['weight'])
            spec.probe_bias = self.probe['bias']
        return spec

    def evaluate(self, setting, split):
        key = f'{split}/{setting_id(setting)}'
        def compute_metrics():
            spec = self.intervention(setting)
            is_math = self.task in {'boxed', 'plain'}
            kind = 'math' if is_math else 'concept'
            def generate(row, dataset_kind):
                messages = []
                if setting['method'] == 'prompting':
                    messages.append({'role': 'system', 'content': BASELINE_PROMPTS[self.task]})
                messages.append({'role': 'user', 'content': row['prompt']})
                return self.store.compute(f'generation/{key}/{dataset_kind}/{row["id"]}', lambda:
                    self.backend.generate(messages, self.config['generation'][dataset_kind], spec,
                                          thinking=self.config['thinking'][dataset_kind]))
            controls, capabilities, truncated = [], [], 0
            for row in self.data[split]:
                output = generate(row, kind)
                truncated += output['finish_reason'] == 'length'
                controls.append(self.store.compute(f'control/{key}/{row["id"]}', lambda:
                    self.judge.control(self.task, row['prompt'], output['response'])))
                if is_math:
                    capabilities.append(self.store.compute(f'accuracy/{key}/{row["id"]}', lambda:
                        self.judge.judge_math(output['response'], row['answer'])['answer_correct']))
            if not is_math:
                for row in self.mmlu[split]:
                    output = generate(row, 'mmlu')
                    capabilities.append(self.store.compute(f'accuracy/{key}/mmlu/{row["id"]}', lambda:
                        parse_answer_letter(output['response']) == row['answer_letter']))
            if not controls or not capabilities:
                raise ValueError('control and capability splits must be nonempty')
            return {'setting': setting, 'control': sum(controls) / len(controls),
                    'capability': sum(capabilities) / len(capabilities),
                    'control_n': len(controls), 'capability_n': len(capabilities),
                    'control_truncated': truncated}
        metrics = self.store.compute(f'metrics/{key}', compute_metrics)
        self.report[split][setting_id(setting)] = metrics
        self.save_report()
        return metrics

    def settings(self, layer):
        groups = {}
        if self.study == 'duration':
            for length in [None, *self.config['duration_lengths']]:
                name = 'additive-full' if length is None else f'additive-prefix-{length}'
                groups[name] = [dict(method=name, operator='additive',
                                     policy='full' if length is None else 'prefix',
                                     length=length or 5, layer=layer, strength=strength)
                                for strength in self.config['additive_strengths']]
            return groups
        for operator, field, strengths in [('additive', 'strength', self.config['additive_strengths']),
                                            ('coast', 'angle', self.config['coast_angles'])]:
            for length in [None, *self.config['prefix_lengths']]:
                policy = 'full' if length is None else 'prefix'
                name = f'{operator}-full' if length is None else f'{operator}-prefix-{length}'
                groups[name] = [dict(method=name, operator=operator, policy=policy,
                                     layer=layer, length=length or 5, **{field: strength}) for strength in strengths]
        if self.alternatives:
            for policy in ('linear', 'exponential', 'das', 'act'):
                strengths = self.config['additive_strengths'] if policy in {'linear', 'exponential'} else [1.]
                groups[policy] = [dict(method=policy, operator='additive', policy=policy, layer=layer,
                                      length=self.config['linear_horizon'] if policy == 'linear' else 5,
                                      strength=strength) for strength in strengths]
        return groups

    def run(self):
        baseline = self.evaluate({'method': 'unsteered'}, 'tune')
        self.capture_directions()
        layer_rows = []
        for layer in self.layers:
            for setting in self.settings(layer)['additive-full']:
                layer_rows.append(self.evaluate(setting, 'tune'))
        chosen = choose_setting(layer_rows, baseline['capability'], self.config['capability_floor'])
        layer_fallback = chosen is None
        if chosen is None:
            chosen = max(layer_rows, key=lambda row: (row['capability'], row['control']))
        layer = chosen['setting']['layer']
        self.report['selected_layer'] = layer
        self.report['layer_selection'] = {'setting': chosen['setting'], 'capability_fallback': layer_fallback}
        self.report['selection'] = {}
        self.evaluate({'method': 'unsteered'}, 'test')
        self.evaluate({'method': 'prompting'}, 'test')
        for name, settings in self.settings(layer).items():
            rows = [self.evaluate(setting, 'tune') for setting in settings]
            selected = choose_setting(rows, baseline['capability'], self.config['capability_floor'])
            if selected is None:
                self.report['selection'][name] = {'status': 'no_feasible_setting'}
            else:
                self.report['selection'][name] = {'status': 'selected', 'setting': selected['setting']}
                if self.study != 'duration':
                    self.evaluate(selected['setting'], 'test')
            if self.study == 'duration':
                for setting in settings:
                    self.evaluate(setting, 'test')
            self.save_report()
        self.report['complete'] = True
        self.save_report()
        return self.report
