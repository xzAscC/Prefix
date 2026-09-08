# AGENTS

1. Use TDD: write tests before writing source code.
2. Start work on a new branch. When the work is finished, commit all changes and open a PR. Keep the PR concise and clear; do not add fluff.
3. Use only the folders already listed. Put each file in its corresponding folder. Do not create new folders unless I am unavailable and you cannot ask me.
4. Use uv to manage virtual environments.
5. If you are unsure, ask me first. If you cannot ask me, search the website or documentation. Do not guess.
6. Save figures as PDF first. Do not save as PNG.
7. Keep code clean and concise.
8. Experiments must log incrementally: persist partial results (JSON) after each completed method or condition, and tee every print to both stdout and a log file under `logs/`. Never buffer all output until the end. `logs/` stores log files only; do not put other files there.
9. All experiment code must support checkpoint/resume: after each completed unit of work (per example, per config, per condition), persist progress to disk (JSON checkpoint); on restart, detect existing checkpoints and skip completed units. Jobs may be preempted at any time; completed work must never be recomputed or lost.
10. Private instructions: if `AGENTS.local.md` exists in the repo root, read it at the start of every session and follow it. It is gitignored; never commit, copy, or expose its contents (or anything referenced by it) in tracked files, commits, or PRs.
11. Wrap long-running experiment entrypoints with `notify_on_exit` from `src/prefix/notify.py` so the owner is emailed on completion or failure (credentials live in gitignored `.env`).
