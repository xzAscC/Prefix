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

## Paper experiments

`configs/paper.yaml` and `scripts/run_paper_experiments.py` are the reproduction
entry point for the experiments in `iclr2027`. The earlier `exp1`–`exp4`
configurations remain available for their original workflows; their grids,
judge versions, and formatting-direction construction are not the paper protocol.

| Paper experiment | Entry point |
| --- | --- |
| Native attention matching geometry and token counts (§4) | `run_section4_bounds.py`, `run_section4_long.py`, `run_section4_native.py`, `run_section4_prompt_diversity.py` |
| OLMo 3 7B behavioral duration and strength (§5) | `run_paper_experiments.py --study duration` |
| Predicted single-token strength β and fixed-query error (§5) | `run_duration_strength.py`, `run_fixed_strength_sweep.py`, `verify_fixed_strength_native.py` |
| Four models × five tasks, Additive and COAST | `run_paper_experiments.py --study models` |
| Linear/exponential decay, DAS and ACT on OLMo 3 7B | Included in the OLMo 3 7B jobs of `--study models` |

The model matrix is Qwen3 1.7B/14B and OLMo 3 7B/32B, at pinned revisions.
Tasks are safety (HarmBench), sentiment (SST-2), politeness
([Intel/polite-guard](https://huggingface.co/datasets/Intel/polite-guard)), and the
boxed/plain MATH-500 final-answer formats. Sentiment and politeness evaluate
negative-class inputs only. Each formatting direction compares its own
instruction against the same neutral problems, using the paper's disjoint
50 direction / 50 tuning / 400 evaluation partition.

```bash
uv sync --group dev
# Inspect all 20 model/task jobs without loading models or calling a judge.
uv run python scripts/run_paper_experiments.py --list
# Run one model/task; omit --model and --task to run the whole matrix.
uv run python scripts/run_paper_experiments.py --model olmo3-7b --task sentiment
# Fixed layer 18 (index 17), lengths 1–15 and full, all four strengths.
uv run python scripts/run_paper_experiments.py --study duration
# Small execution check, kept separate from full results.
uv run python scripts/run_paper_experiments.py --model olmo3-7b --task safety \
  --run-id smoke --limit 2 --layer 17
```

The judge uses the configured Gemini model through Vertex AI with Application
Default Credentials and `GOOGLE_CLOUD_PROJECT`. The default model name follows
the paper (`gemini-3.5-flash`); model availability and access must be provided by
the execution environment. A changed judge is a protocol change and is recorded
in the manifest, not substituted silently. Generation limits are 512 tokens for
concept control, 1024 for MMLU-Pro, and 4096 for MATH-500. Decode is greedy.

Every completed activation capture, generation, and score is saved separately as
an atomic JSON checkpoint. Restart the identical command to resume: a judge
failure does not repeat a completed generation. Configuration, code digest,
model revision, and smoke limits are bound to the checkpoint manifest. Use a new
`--run-id` after changing them. Partial summaries are updated after every
condition; stdout is also written immediately to `logs/`. Checkpoints and
results stay in their respective ignored directories.

Implementation choices not specified in the manuscript are explicit in
`configs/paper.yaml`: 100 direction examples per class, 50 tuning examples,
ten interior layer candidates, a linear horizon of 128, and thinking enabled
for MMLU-Pro/MATH-500. For SST-2/PoliteGuard, tuning uses unused negative training
examples; the published evaluation split is reserved for testing. Safety uses
50 HarmBench prompts for tuning and the remaining 350 for testing. Both math
formats share the same seeded 50/50/400 partition.

The shared layer is selected with additive/full validation results and then
held fixed across operators and policies. Selection maximizes control subject
to 90% of unsteered capability. If no layer meets that floor, layer selection
uses the highest-capability candidate and reports the fallback; each subsequent
method with no feasible strength is reported as such, without a fabricated
selected result. Strength selection never uses test scores. The duration study
also reports every test strength to support the strength trade-off plot.

DAS uses the union of top-p nucleus supports (`p=0.9`), the paper's capped KL
rule (`c_max=2`), and isolated cache copies for its two probe forwards. ACT uses
a balanced linear logistic classifier trained on direction-construction
activations, L2 regularization 0.01, and the paper's `A=12, beta=0` coefficient.
These three implementation choices—shared-layer selection, DAS top-p, and ACT
probe fitting—are configurable. COAST solves the stated weighted quadratic
objective with norm and target-cosine constraints using the uncentered reference
second moment, including the degenerate trust-region case. Its eigendecomposition
is reused across tokens. The Transformers backend supports both model families;
DAS uses extra forwards and temporary cache copies and therefore needs more
memory and compute than constant steering.

The code includes numerical operator checks, tiny Qwen3/OLMo3 cache tests, and
five-task orchestration/resume tests. These checks do not establish full-model
experimental results. Generated notebooks, figures, datasets, and results are
excluded from commits; their directories retain only `.gitkeep`.

Section 4 extraction now records OLMo's all-head Q/K normalization, the model's
actual RoPE/YaRN frequencies and scaling, and sliding-window limits. Both head
replay and fixed-context replay are checked against native OLMo3 attention.
Caches from the earlier Qwen-style extraction are incompatible (native schema
2); use a fresh experiment workspace and repeat extraction rather than mixing
them with corrected measurements. Typical Section 4 phase order is:

```bash
uv run python scripts/run_section4_bounds.py extract
uv run python scripts/run_section4_bounds.py analyze
uv run python scripts/run_section4_long.py extract
uv run python scripts/run_section4_native.py prepare
uv run python scripts/run_section4_native.py extract
uv run python scripts/run_section4_native.py analyze
```

These legacy attention scripts expect the pinned model in the local Hugging
Face cache. Their results concern frozen attention states, separately from the
free-generation control/capability results of `run_paper_experiments.py`.
