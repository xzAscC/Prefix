# Prefix

Source code for Prefix.

Requires Python 3.13+. Use [uv](https://docs.astral.sh/uv/) for the virtual environment.

```bash
uv sync --group dev
uv run pytest
```

## Files Architecture

```text
src/: source code
tests/: test code
notebooks/: jupyter notebooks
scripts/: sh files to run the code
logs/: log for different results
figs/: figs for experiments
data/: store different datasets and other data
configs/: yaml files for default configs
checkpoints/: store different checkpoints if has
README.md: this file
AGENTS.md: agent rules
pyproject.toml: project metadata and build config
.python-version: Python version pin
.gitignore: git ignore rules
.ignore: ! un-ignore gitignored paths so agents can find them
LICENSE: MIT license text
```

## License

MIT. See [LICENSE](LICENSE).
