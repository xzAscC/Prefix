import pytest
import torch

from prefix.paper_experiments import PaperExperiment
from prefix.paper_protocol import Checkpoint


class Backend:
    layers = [None, None]
    device = torch.device('cpu')
    def __init__(self):
        self.captures = 0
        self.generations = 0
        self.policies = set()
    def capture(self, messages, layers, **kwargs):
        self.captures += 1
        sign = 1. if messages[-1]['content'].startswith('positive') else -1.
        return {layer: torch.tensor([sign, 1., .5]) for layer in layers}
    def generate(self, messages, max_tokens, spec=None, **kwargs):
        self.generations += 1
        if spec:
            self.policies.add((spec.operator, spec.policy, spec.length))
        return dict(response='The answer is (A)', token_ids=[1], coefficients=[0.], finish_reason='eos')


class Judge:
    def __init__(self):
        self.fail = False
    def control(self, *args):
        if self.fail:
            raise RuntimeError('temporary scoring interruption')
        return True
    def judge_math(self, response, answer):
        return {'answer_correct': True}


def config():
    return dict(additive_strengths=[.1], coast_angles=[30], prefix_lengths=[1, 5],
                capability_floor=.9, linear_horizon=128, exponential_tau=128,
                das_top_p=.9, das_maximum=2., act_amplitude=12., act_bias=0.,
                probe_regularization=.01, probe_max_iter=30,
                generation=dict(concept=512, mmlu=1024, math=4096),
                thinking=dict(concept=False, mmlu=True, math=True))


@pytest.mark.parametrize('task', ['safety', 'sentiment', 'politeness', 'boxed', 'plain'])
def test_full_protocol_covers_methods_and_resumes_without_repeating_work(tmp_path, task):
    backend, judge = Backend(), Judge()
    data = dict(positive=['positive 0', 'positive 1'], negative=['negative 0', 'negative 1'],
                tune=[dict(id='t0', prompt='tune', answer='A')],
                test=[dict(id='e0', prompt='test', answer='A')])
    mmlu = {split: [dict(id=split, prompt='multiple choice', answer_letter='A')] for split in ['tune', 'test']}
    store = Checkpoint(tmp_path, 'job', {'task': task})
    run = PaperExperiment(config(), task, backend, judge, store, data, mmlu,
                         layers=[0, 1], alternatives=True, summary_path=tmp_path / 'summary.json')
    report = run.run()
    assert report['selected_layer'] in [0, 1]
    assert len(report['test']) == 12  # two baselines, six operator/duration methods, four policies
    assert ('additive', 'act', 5) in backend.policies
    assert ('additive', 'das', 5) in backend.policies
    assert ('coast', 'prefix', 1) in backend.policies
    assert ('coast', 'prefix', 5) in backend.policies
    previous = backend.captures, backend.generations
    assert run.run() == report
    assert (backend.captures, backend.generations) == previous


def test_generation_survives_scoring_interruption(tmp_path):
    backend, judge = Backend(), Judge()
    judge.fail = True
    data = dict(positive=['positive'], negative=['negative'],
                tune=[dict(id='t', prompt='p')], test=[dict(id='e', prompt='q')])
    mmlu = {split: [dict(id=split, prompt='mcq', answer_letter='A')] for split in ['tune', 'test']}
    run = PaperExperiment(config(), 'safety', backend, judge, Checkpoint(tmp_path, 'r', {}),
                         data, mmlu, layers=[0], alternatives=False, summary_path=tmp_path/'s.json')
    with pytest.raises(RuntimeError):
        run.evaluate({'method': 'unsteered'}, 'tune')
    assert backend.generations == 1
    judge.fail = False
    run.evaluate({'method': 'unsteered'}, 'tune')
    assert backend.generations == 2  # saved control response reused; only MMLU generated now


def test_duration_study_keeps_layer_fixed_and_evaluates_all_strengths(tmp_path):
    cfg = config()
    cfg['duration_lengths'] = [1, 2, 5, 15]
    cfg['additive_strengths'] = [.01, .1, 1., 10.]
    backend, judge = Backend(), Judge()
    data = dict(positive=['positive'], negative=['negative'],
                tune=[dict(id='t', prompt='p')], test=[dict(id='e', prompt='q')])
    mmlu = {split: [dict(id=split, prompt='mcq', answer_letter='A')] for split in ['tune', 'test']}
    run = PaperExperiment(cfg, 'safety', backend, judge, Checkpoint(tmp_path, 'duration', {}),
                         data, mmlu, layers=[1], alternatives=False, summary_path=tmp_path/'s.json', study='duration')
    report = run.run()
    assert report['selected_layer'] == 1
    assert len(report['test']) == 22  # (four prefix lengths + full) x four strengths + baselines
    assert all(row['setting'].get('operator', 'additive') == 'additive' for row in report['test'].values())
