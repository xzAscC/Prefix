from pathlib import Path
import yaml
from prefix.paper_cli import jobs, candidate_layers, main

ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration_covers_exact_model_task_matrix():
    cfg = yaml.safe_load((ROOT/'configs/paper.yaml').read_text())
    matrix = jobs(cfg, 'all', 'all', 'models')
    assert len(matrix) == 20
    assert {model for model, task in matrix} == {'qwen3-1.7b', 'qwen3-14b', 'olmo3-7b', 'olmo3-32b'}
    assert jobs(cfg, 'all', 'all', 'duration') == [('olmo3-7b', 'safety')]
    assert candidate_layers(32, cfg['protocol']['layer_fractions']) == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]


def test_list_is_offline_and_reports_all_jobs(capsys):
    main(['--config', str(ROOT/'configs/paper.yaml'), '--list'])
    output = capsys.readouterr().out
    assert output.count('models:') == 20
    assert 'qwen3-1.7b / politeness' in output


def test_cli_writes_logs_results_and_resumes_with_manifest(tmp_path, monkeypatch):
    import json
    import torch
    import prefix.paper_cli as cli
    class Backend:
        layers = [None, None, None]
        device = torch.device('cpu')
        generations = 0
        def capture(self, messages, layers, **kwargs):
            sign = 1. if messages[-1]['content'] == 'positive' else -1.
            return {layer: torch.tensor([sign, 1.]) for layer in layers}
        def generate(self, *args, **kwargs):
            self.generations += 1
            return dict(response='The answer is (A)', token_ids=[1], coefficients=[0.], finish_reason='eos')
    class Judge:
        def __init__(self, **kwargs):
            pass
        def _call(self, prompt):
            return 'SAFE'
        def judge_math(self, response, answer):
            return {'answer_correct': True}
    backend = Backend()
    monkeypatch.setattr(cli.PaperBackend, 'load', lambda *args: backend)
    monkeypatch.setattr(cli, 'GeminiJudge', Judge)
    monkeypatch.setattr(cli, 'prepare_task', lambda *args, **kwargs: dict(
        positive=['positive'], negative=['negative'],
        tune=[dict(id='t', prompt='tune')], test=[dict(id='e', prompt='test')]))
    monkeypatch.setattr(cli, 'load_mmlu_pro', lambda split: [dict(question='q', options=['yes', 'no'], answer_letter='A')])
    cfg = yaml.safe_load((ROOT/'configs/paper.yaml').read_text())
    cfg['protocol']['additive_strengths'] = [.1]
    cfg['protocol']['coast_angles'] = [30]
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    args = ['--config', str(path), '--root', str(tmp_path), '--model', 'olmo3-7b',
            '--task', 'safety', '--limit', '1', '--layer', '1']
    cli.main(args)
    calls = backend.generations
    cli.main(args)
    assert backend.generations == calls
    report = json.loads((tmp_path/'results/paper.models.olmo3-7b.safety.json').read_text())
    assert report['complete'] and report['metadata']['smoke_limit'] == 1
    log = (tmp_path/'logs/paper.models.olmo3-7b.safety.log').read_text()
    assert 'completed generation/' in log and log.count('Completed paper.models.olmo3-7b.safety') == 2
