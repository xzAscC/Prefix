import os
from pathlib import Path
import subprocess


def test_batch_stops_on_failed_extraction_and_never_emails(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/sdsc_section4_native.sbatch'
    uv = tmp_path / 'uv'
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nexit 17\n')
    uv.chmod(0o755)
    calls = tmp_path / 'calls'
    env = {**os.environ,'PATH':f'{tmp_path}:{os.environ["PATH"]}',
           'SLURM_SUBMIT_DIR':str(tmp_path),'CALLS':str(calls)}
    result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode == 17
    assert len(calls.read_text().splitlines()) == 1
    assert 'extract' in calls.read_text()
    assert 'mail' not in script.read_text().lower()
