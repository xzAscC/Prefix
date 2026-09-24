import os
from pathlib import Path
import subprocess


def test_batch_runs_only_paper_experiment_phases_and_never_emails(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/sdsc_section4_native.sbatch'
    uv = tmp_path / 'uv'
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    uv.chmod(0o755)
    calls = tmp_path / 'calls'
    env = {**os.environ,'PATH':f'{tmp_path}:{os.environ["PATH"]}',
           'SLURM_SUBMIT_DIR':str(tmp_path),'CALLS':str(calls)}
    result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode == 0
    commands = calls.read_text().splitlines()
    assert len(commands) == 2
    assert commands[0].endswith('run_section4_native.py extract')
    assert commands[1].endswith('run_section4_native.py analyze')
    assert 'plot_section4_native.py' not in script.read_text()
    assert 'mail' not in script.read_text().lower()
