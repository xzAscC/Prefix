from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from math import exp
from pathlib import Path
from typing import Any

import pytest

from prefix import no_steering
from prefix.judge import JudgeBlocked, JudgeParseError, JudgeRetryableError


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_no_steering.py"
MODEL_IDS = tuple(no_steering.MODEL_MATRIX)
PINNED_MODELS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "allenai/Olmo-3-7B-Think": "d97e442d7cc678210054dbcc9b440894d62c89a4",
    "allenai/Olmo-3-32B-Think": "f2edda15216e738ef2bb73771e11890e152b2112",
}


def entrypoint():
    if not SCRIPT.exists():
        pytest.fail(f"missing standalone evaluator entrypoint: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("run_no_steering", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_benchmark_records_and_exact_pinned_models_are_selected() -> None:
    runner = entrypoint()
    records: dict[str, list[dict[str, object]]] = {
        benchmark: [
            {"id": f"{benchmark}-{index}", "value": index}
            for index in range(int(runner.DATASET_SCOPES[benchmark]["count"]))
        ]
        for benchmark in runner.DATASET_SCOPES
    }
    calls: list[tuple[str, str, str, int]] = []

    def load(benchmark: str, *, split: str, count: int) -> list[dict[str, object]]:
        calls.append((benchmark, split, str(count), len(records[benchmark])))
        return records[benchmark]

    plan = runner.load_benchmarks(load=load)

    assert [item[0] for item in calls] == ["harmbench", "mmlu_pro", "math500"]
    assert [(item[1], int(item[2])) for item in calls] == [
        ("all", 400),
        ("test", 12032),
        ("test", 500),
    ]
    assert [item["id"] for item in plan["harmbench"][:2]] == [
        "harmbench-0",
        "harmbench-1",
    ]
    assert len(plan["harmbench"]) == 400
    assert len(plan["mmlu_pro"]) == 12032
    assert len(plan["math500"]) == 500
    assert tuple(runner.MODEL_MATRIX) == MODEL_IDS
    for model_id, revision in PINNED_MODELS.items():
        assert runner.model_spec(model_id) == no_steering.model_spec(model_id)
        assert runner.model_spec(model_id).revision == revision


def test_direct_script_preflight_imports_helper_before_snapshot_check(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-id",
            MODEL_IDS[0],
            "--phase",
            "preflight",
            "--cache-root",
            str(tmp_path),
        ],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError: No module named 'scripts'" not in completed.stderr
    assert MODEL_IDS[0] in completed.stderr


@pytest.mark.parametrize(
    ("benchmark", "delta"),
    [("harmbench", -1), ("mmlu_pro", 1), ("math500", -1)],
)
def test_load_benchmarks_rejects_incomplete_or_oversized_scopes(
    benchmark: str, delta: int
) -> None:
    runner = entrypoint()
    records: dict[str, list[dict[str, object]]] = {
        name: [
            {"id": f"{name}-{index}"}
            for index in range(int(runner.DATASET_SCOPES[name]["count"]))
        ]
        for name in runner.DATASET_SCOPES
    }
    if delta < 0:
        records[benchmark] = records[benchmark][: len(records[benchmark]) + delta]
    else:
        records[benchmark].append({"id": f"{benchmark}-extra"})

    with pytest.raises(ValueError, match=f"{benchmark}.*expected"):
        runner.load_benchmarks(
            load=lambda name, **kwargs: records[name],
        )


def test_load_benchmarks_rejects_explicit_duplicate_ids() -> None:
    runner = entrypoint()
    records: dict[str, list[dict[str, object]]] = {
        name: [
            {"value": index}
            for index in range(int(runner.DATASET_SCOPES[name]["count"]))
        ]
        for name in runner.DATASET_SCOPES
    }
    records["harmbench"][0]["id"] = "same"
    records["harmbench"][1]["id"] = "same"

    with pytest.raises(ValueError, match="harmbench.*duplicate"):
        runner.load_benchmarks(load=lambda name, **kwargs: records[name])


def test_load_benchmarks_allows_identical_no_id_rows_with_deterministic_unique_ids() -> (
    None
):
    runner = entrypoint()
    records = {
        name: [
            {"value": index}
            for index in range(int(runner.DATASET_SCOPES[name]["count"]))
        ]
        for name in runner.DATASET_SCOPES
    }
    records["harmbench"][1] = dict(records["harmbench"][0])

    first = runner.load_benchmarks(load=lambda name, **kwargs: records[name])
    second = runner.load_benchmarks(load=lambda name, **kwargs: records[name])
    first_ids = [str(row["id"]) for row in first["harmbench"]]
    second_ids = [str(row["id"]) for row in second["harmbench"]]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first_ids[0] != first_ids[1]


def test_load_benchmarks_generates_deterministic_ids_for_rows_without_ids() -> None:
    runner = entrypoint()
    records = {
        name: [
            {"value": index}
            for index in range(int(runner.DATASET_SCOPES[name]["count"]))
        ]
        for name in runner.DATASET_SCOPES
    }

    first = runner.load_benchmarks(load=lambda name, **kwargs: records[name])
    second = runner.load_benchmarks(load=lambda name, **kwargs: records[name])
    assert first["math500"][0]["id"] == second["math500"][0]["id"]
    assert str(first["math500"][0]["id"]).startswith("math500-")


def test_benchmark_manifest_distinguishes_full_and_intentionally_limited_runs() -> None:
    runner = entrypoint()
    records = {
        name: [
            {"id": f"{name}-{index}"}
            for index in range(int(runner.DATASET_SCOPES[name]["count"]))
        ]
        for name in runner.DATASET_SCOPES
    }
    plan = runner.load_benchmarks(load=lambda name, **kwargs: records[name])

    full = runner.benchmark_manifest(plan)
    partial = runner.benchmark_manifest(
        {name: values[:2] for name, values in plan.items()}, limited=True
    )
    assert full["mmlu_pro"] == {
        "split": "test",
        "expected_count": 12032,
        "loaded_count": 12032,
        "complete": True,
        "limited": False,
    }
    assert partial["mmlu_pro"] == {
        "split": "test",
        "expected_count": 12032,
        "loaded_count": 2,
        "complete": False,
        "limited": True,
    }


def test_raw_response_logprob_schema_and_response_only_ppl_are_preserved(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    raw_response = {
        "text": "The answer is (B)",
        "prompt_token_ids": [11, 12, 13],
        "outputs": [
            {
                "token_ids": [21, 22],
                "logprobs": [
                    {"token": 21, "logprob": -0.5},
                    {"token": 22, "logprob": -1.0},
                ],
            }
        ],
    }
    record = {
        "id": "mmlu-0",
        "benchmark": "mmlu_pro",
        "prompt": "Choose B",
        "gold": "B",
        "metadata": {"category": "demo"},
    }

    rows = runner.evaluate_records(
        [record],
        model_id=MODEL_IDS[0],
        generate=lambda item: raw_response,
        output_path=tmp_path / "responses.jsonl",
        extract_answer=lambda text: text.rsplit("(", 1)[1][0],
    )

    assert rows == [
        {
            "id": "mmlu-0",
            "benchmark": "mmlu_pro",
            "model_id": MODEL_IDS[0],
            "prompt": "Choose B",
            "raw_response": raw_response,
            "gold": "B",
            "extracted_answer": "B",
            "selected_generated_token_logprobs": [-0.5, -1.0],
            "prompt_token_count": 3,
            "generated_token_count": 2,
            "status": "ok",
            "error": None,
            "metadata": {"category": "demo"},
        }
    ]
    assert runner.response_only_ppl(rows) == pytest.approx(exp(1.5 / 2))
    assert runner.response_only_ppl(
        [
            {
                "selected_generated_token_logprobs": [-0.5, -1.0],
                "prompt_token_logprobs": [-1000.0, -1000.0],
            }
        ]
    ) == pytest.approx(exp(1.5 / 2))
    saved = json.loads((tmp_path / "responses.jsonl").read_text().splitlines()[0])
    assert saved["prompt"] == "Choose B"
    assert saved["raw_response"] == raw_response


def test_vllm_nested_logprobs_select_token_id_and_validate_lengths() -> None:
    runner = entrypoint()
    raw_response = {
        "outputs": [
            {
                "token_ids": [21, 22],
                "logprobs": [
                    {21: {"logprob": -0.5, "rank": 1}, 99: {"logprob": -9.0}},
                    {22: {"logprob": -1.0, "rank": 1}},
                ],
            }
        ]
    }
    assert runner._selected_logprobs(raw_response) == [-0.5, -1.0]
    with pytest.raises(ValueError, match="logprob length"):
        runner._selected_logprobs(
            {
                "outputs": [
                    {"token_ids": [21, 22], "logprobs": [{21: {"logprob": -0.5}}]}
                ]
            }
        )


def test_real_like_logprob_objects_are_json_native_and_selected() -> None:
    runner = entrypoint()

    class Logprob:
        def __init__(self, value: float) -> None:
            self.logprob = value
            self.rank = 1

    assert runner._selected_logprobs(
        {
            "outputs": [
                {
                    "token_ids": [21, 22],
                    "logprobs": [{21: Logprob(-0.5)}, {22: Logprob(-1.0)}],
                }
            ]
        }
    ) == [-0.5, -1.0]


def test_batched_evaluation_preserves_order_and_exact_batch_sizes(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    records = [
        {"id": f"r{index}", "prompt": f"p{index}", "benchmark": "math500"}
        for index in range(5)
    ]
    calls: list[list[str]] = []

    def generate(prompts: list[str], params: object) -> list[dict[str, object]]:
        calls.append(prompts)
        return [
            {"text": prompt, "token_ids": [index]}
            for index, prompt in enumerate(prompts)
        ]

    rows = runner.evaluate_records_batched(
        records,
        model_id=MODEL_IDS[0],
        generate=generate,
        output_path=tmp_path / "responses.jsonl",
        batch_size=2,
        sampling_params=object(),
    )
    assert calls == [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert [row["id"] for row in rows] == ["r0", "r1", "r2", "r3", "r4"]


def test_batched_evaluation_persists_one_durable_append_per_completed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    records = [
        {"id": f"r{index}", "prompt": f"p{index}", "benchmark": "math500"}
        for index in range(5)
    ]
    appends: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "append_jsonl",
        lambda path, rows: appends.append(list(rows)),
    )

    runner.evaluate_records_batched(
        records,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: [
            {"text": prompt, "token_ids": [1]} for prompt in prompts
        ],
        output_path=tmp_path / "responses.jsonl",
        batch_size=2,
        sampling_params=object(),
    )
    assert [[row["id"] for row in batch] for batch in appends] == [
        ["r0", "r1"],
        ["r2", "r3"],
        ["r4"],
    ]


def test_batched_evaluation_rejects_mismatched_output_count(tmp_path: Path) -> None:
    runner = entrypoint()
    records = [{"id": "r0", "prompt": "p0", "benchmark": "math500"}]
    rows = runner.evaluate_records_batched(
        records,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: [],
        output_path=tmp_path / "responses.jsonl",
        batch_size=2,
        sampling_params=object(),
    )
    assert rows[0]["status"] == "error"
    assert "output count" in str(rows[0]["error"])


def test_batched_generation_retry_replaces_prior_error_row(tmp_path: Path) -> None:
    runner = entrypoint()
    output = tmp_path / "batched.jsonl"
    record = [{"id": "r0", "prompt": "p0", "benchmark": "math500"}]
    runner.evaluate_records_batched(
        record,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: (_ for _ in ()).throw(RuntimeError("first")),
        output_path=output,
        batch_size=1,
        sampling_params=object(),
    )
    runner.evaluate_records_batched(
        record,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: [{"text": "ok", "token_ids": [1]}],
        output_path=output,
        batch_size=1,
        sampling_params=object(),
        retry_errors=True,
    )
    saved = runner.read_jsonl(output)
    assert len(saved) == 1
    assert saved[0]["status"] == "ok"


def test_batched_generation_retry_uses_one_repair_rewrite_for_many_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    output = tmp_path / "batched-many.jsonl"
    records = [
        {"id": f"r{index}", "prompt": f"p{index}", "benchmark": "math500"}
        for index in range(8)
    ]
    runner.evaluate_records_batched(
        records,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: (_ for _ in ()).throw(RuntimeError("first")),
        output_path=output,
        batch_size=2,
        sampling_params=object(),
    )
    rewrites: list[int] = []
    original_rewrite = runner._rewrite_jsonl

    def tracked_rewrite(path: str | Path, rows: list[dict[str, object]]) -> None:
        rewrites.append(len(rows))
        original_rewrite(path, rows)

    monkeypatch.setattr(runner, "_rewrite_jsonl", tracked_rewrite)
    runner.evaluate_records_batched(
        records,
        model_id=MODEL_IDS[0],
        generate=lambda prompts, params: [
            {"text": prompt, "token_ids": [1]} for prompt in prompts
        ],
        output_path=output,
        batch_size=2,
        sampling_params=object(),
        retry_errors=True,
    )
    assert len(rewrites) == 1


def test_partial_logprob_coverage_is_error_and_ppl_is_null() -> None:
    runner = entrypoint()
    rows = [
        {
            "id": "x",
            "status": "ok",
            "generated_token_count": 2,
            "selected_generated_token_logprobs": [-1.0],
        }
    ]
    assert runner.ppl_summary(rows)["ppl"] is None
    assert runner.ppl_summary(rows)["coverage_ratio"] == 0.5


def test_manifest_binds_identity_ids_roots_and_generation_contract(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    records = {name: [{"id": f"{name}-0"}] for name in runner.DATASET_SCOPES}
    manifest = runner.checkpoint_manifest(
        records,
        model_id=MODEL_IDS[0],
        sampling={"temperature": 0.0, "logprobs": 1},
        batch_prompts=4,
        max_model_len=8192,
        output_roots={
            "checkpoint": str(tmp_path / "checkpoints"),
            "output": str(tmp_path / "results"),
        },
        limited=True,
    )
    assert manifest["model_slug"] == "Qwen--Qwen3-4B"
    assert manifest["model_revision"] == PINNED_MODELS[MODEL_IDS[0]]
    assert manifest["benchmark_ids"]["math500"] == ["math500-0"]
    assert manifest["limited"] is True


def test_score_phase_does_not_construct_gemini_for_mmlu_only(tmp_path: Path) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        path,
        [
            {
                "id": "m0",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
            }
        ],
    )
    calls: list[bool] = []
    runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: calls.append(True),
        limited=True,
    )
    assert calls == []


def test_score_resume_retries_transient_errors_but_not_terminal_statuses(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "scores.jsonl"
    runner.append_jsonl(
        output,
        [
            {"id": "transient", "status": "error", "error": "HTTP 503"},
            {"id": "blocked", "status": "blocked", "error": "blocked"},
            {"id": "unparseable", "status": "unparseable", "error": "bad JSON"},
        ],
    )
    rows = [
        {
            "id": "transient",
            "benchmark": "mmlu_pro",
            "gold": "A",
            "extracted_answer": "A",
            "status": "ok",
        }
    ]
    written = runner.score_responses(rows, output_path=output, judge=object())
    assert [row["id"] for row in written] == ["transient"]


def test_formal_completeness_requires_expected_denominators() -> None:
    runner = entrypoint()
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_score_completeness(
            [{"benchmark": "mmlu_pro", "status": "ok"}], limited=False
        )


def test_resume_repairs_truncated_last_line_and_keeps_error_context(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "responses.jsonl"
    output.write_text('{"id":"done","status":"ok"}\n{"id":"cut"', encoding="utf-8")
    calls: list[str] = []

    def generate(record: dict[str, Any]) -> dict[str, object]:
        calls.append(str(record["id"]))
        raise RuntimeError("boom")

    rows = runner.evaluate_records(
        [
            {"id": "done", "prompt": "old", "gold": "A"},
            {"id": "new", "prompt": "p", "gold": "B"},
        ],
        model_id=MODEL_IDS[0],
        generate=generate,
        output_path=output,
    )
    assert calls == ["new"]
    assert rows[0]["prompt"] == "p"
    assert rows[0]["gold"] == "B"
    assert rows[0]["raw_response"] is None
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_scores_and_atomic_aggregate_artifacts_include_coverage_and_provenance(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    rows = [
        {
            "id": "m",
            "benchmark": "mmlu_pro",
            "gold": "A",
            "extracted_answer": "A",
            "status": "ok",
            "selected_generated_token_logprobs": [-1.0],
        },
        {
            "id": "x",
            "benchmark": "math500",
            "gold": "42",
            "judge": {"answer_correct": True},
            "status": "ok",
            "selected_generated_token_logprobs": [-2.0, -1.0],
        },
        {
            "id": "h",
            "benchmark": "harmbench",
            "gemini_label": "unsafe",
            "status": "ok",
            "selected_generated_token_logprobs": [],
        },
    ]
    scores = runner.score_records(rows)
    assert scores["mmlu_pro"] == {"accuracy": 1.0, "correct": 1, "count": 1}
    assert scores["math500"] == {"accuracy": 1.0, "correct": 1, "count": 1}
    assert scores["harmbench"]["label"] == "unsafe"
    assert scores["harmbench"]["canonical"] is False
    summary = runner.aggregate_records(rows, model_id=MODEL_IDS[0])
    assert summary["ppl"]["selected_token_count"] == 3
    assert summary["ppl"]["covered_records"] == 2
    assert summary["provenance"]["revision"] == PINNED_MODELS[MODEL_IDS[0]]
    paths = runner.write_aggregate_artifacts(tmp_path, summary)
    assert json.loads(paths["summary"].read_text(encoding="utf-8")) == summary


def test_completed_ids_skip_generation_and_failures_are_isolated(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    calls: list[str] = []

    def generate(record: dict[str, Any]) -> dict[str, object]:
        calls.append(record["id"])
        if record["id"] == "bad":
            raise RuntimeError("synthetic generation failure")
        return {"text": f"answer-{record['id']}", "token_ids": [1]}

    rows = runner.evaluate_records(
        [
            {"id": "done", "prompt": "old", "gold": "A"},
            {"id": "bad", "prompt": "fails", "gold": "B"},
            {"id": "later", "prompt": "runs", "gold": "C"},
        ],
        model_id=MODEL_IDS[1],
        generate=generate,
        output_path=tmp_path / "resume.jsonl",
        completed_ids={"done"},
    )

    assert calls == ["bad", "later"]
    assert [row["id"] for row in rows] == ["bad", "later"]
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "synthetic generation failure"
    assert rows[1]["status"] == "ok"


def test_retry_errors_reprocesses_error_rows_but_default_resume_skips_them(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "retry.jsonl"
    runner.evaluate_records(
        [{"id": "x", "prompt": "p", "gold": "A"}],
        model_id=MODEL_IDS[0],
        generate=lambda _: (_ for _ in ()).throw(RuntimeError("first")),
        output_path=output,
    )
    calls: list[str] = []
    runner.evaluate_records(
        [{"id": "x", "prompt": "p", "gold": "A"}],
        model_id=MODEL_IDS[0],
        generate=lambda record: calls.append(str(record["id"])) or {"text": "ok"},
        output_path=output,
    )
    assert calls == []
    runner.evaluate_records(
        [{"id": "x", "prompt": "p", "gold": "A"}],
        model_id=MODEL_IDS[0],
        generate=lambda record: calls.append(str(record["id"])) or {"text": "ok"},
        output_path=output,
        retry_errors=True,
    )
    assert calls == ["x"]


def test_generation_retry_compacts_prior_error_without_duplicate_authoritative_id(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "retry.jsonl"
    runner.evaluate_records(
        [{"id": "x", "prompt": "p", "gold": "A"}],
        model_id=MODEL_IDS[0],
        generate=lambda _: (_ for _ in ()).throw(RuntimeError("first")),
        output_path=output,
    )
    runner.evaluate_records(
        [{"id": "x", "prompt": "p", "gold": "A"}],
        model_id=MODEL_IDS[0],
        generate=lambda _: {"text": "ok", "token_ids": [1]},
        output_path=output,
        retry_errors=True,
    )
    saved = runner.read_jsonl(output)
    assert [row["id"] for row in saved] == ["x"]
    assert saved[0]["status"] == "ok"


def test_benchmark_failure_does_not_stop_next_benchmark() -> None:
    runner = entrypoint()
    completed: list[str] = []

    def evaluate(
        benchmark: str, records: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        completed.append(benchmark)
        if benchmark == "harmbench":
            raise RuntimeError("synthetic benchmark failure")
        return [{"id": str(records[0]["id"]), "status": "ok"}]

    result = runner.evaluate_benchmarks(
        {
            "harmbench": [{"id": "h0"}],
            "mmlu_pro": [{"id": "m0"}],
            "math500": [{"id": "x0"}],
        },
        evaluate=evaluate,
    )

    assert completed == ["harmbench", "mmlu_pro", "math500"]
    assert result["harmbench"] == [
        {"id": "harmbench", "status": "error", "error": "synthetic benchmark failure"}
    ]
    assert result["mmlu_pro"] == [{"id": "m0", "status": "ok"}]
    assert result["math500"] == [{"id": "x0", "status": "ok"}]


def test_gemini_harmbench_label_is_noncanonical() -> None:
    runner = entrypoint()
    assert runner.harmbench_label("refusal") == "refusal"
    assert runner.harmbench_label("compliance") == "compliance"
    assert runner.harmbench_label("unsafe") == "unsafe"
    assert runner.harmbench_label("safe") == "safe"
    assert runner.harmbench_label("YES") == "YES"
    assert runner.harmbench_label("NO") == "NO"
    assert runner.harmbench_label("unknown") == "unknown"
    assert runner.harmbench_label("harmful") == "harmful"
    assert runner.harmbench_label("harmless") == "harmless"


def test_score_responses_is_incremental_resumable_and_counts_failures(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    responses = [
        {
            "id": "h-ok",
            "benchmark": "harmbench",
            "prompt": "harm request",
            "raw_response": {"text": "refused"},
            "status": "ok",
            "metadata": {"behavior": "harm request"},
        },
        {
            "id": "h-blocked",
            "benchmark": "harmbench",
            "prompt": "blocked",
            "raw_response": {"text": "answer"},
            "status": "ok",
            "metadata": {},
        },
        {
            "id": "mmlu-bad",
            "benchmark": "mmlu_pro",
            "prompt": "question",
            "extracted_answer": "B",
            "gold": "A",
            "status": "ok",
            "metadata": {},
        },
        {
            "id": "math-unparseable",
            "benchmark": "math500",
            "prompt": "math",
            "raw_response": {"text": "answer"},
            "gold": "42",
            "status": "ok",
            "metadata": {},
        },
        {
            "id": "generation-error",
            "benchmark": "math500",
            "prompt": "failed",
            "gold": "42",
            "status": "error",
            "error": "generation failed",
            "metadata": {},
        },
    ]

    class FakeJudge:
        safety_calls: list[tuple[str, str]] = []
        math_calls: list[tuple[str, str]] = []

        def judge_safety(self, request: str, response: str) -> bool:
            self.safety_calls.append((request, response))
            if request == "blocked":
                raise JudgeBlocked("blocked")
            return True

        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            self.math_calls.append((response, expected))
            raise JudgeParseError(
                "Gemini math judge returned an unparseable JSON result"
            )

    output = tmp_path / "scores.jsonl"
    fake_judge = FakeJudge()
    first = runner.score_responses(responses, output_path=output, judge=fake_judge)
    assert len(first) == 5
    saved = runner.read_jsonl(output)
    assert len(saved) == 5
    assert saved[0]["gemini_label"] == "safe"
    assert saved[0]["canonical"] is False
    assert saved[1]["status"] == "blocked"
    assert saved[3]["status"] == "unparseable"
    assert saved[4]["status"] == "error"
    assert fake_judge.safety_calls == [
        ("harm request", "refused"),
        ("blocked", "answer"),
    ]
    assert fake_judge.math_calls == [("answer", "42")]

    class ExplodingJudge:
        def judge_safety(self, request: str, response: str) -> bool:
            raise AssertionError("completed score was regenerated")

        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise AssertionError("completed score was regenerated")

    assert (
        runner.score_responses(responses, output_path=output, judge=ExplodingJudge())
        == []
    )
    summary = runner.score_summary(saved)
    assert summary["harmbench"]["denominator"] == 2
    assert summary["harmbench"]["status_counts"] == {"blocked": 1, "ok": 1}
    assert summary["mmlu_pro"]["denominator"] == 1
    assert summary["math500"]["denominator"] == 2
    assert summary["math500"]["status_counts"]["error"] == 1


def test_score_retry_compacts_transient_error_and_preserves_terminal_rows(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "scores.jsonl"
    runner.append_jsonl(
        output,
        [
            {"id": "retry", "status": "error", "error": "HTTP 503"},
            {"id": "blocked", "status": "blocked", "error": "blocked"},
        ],
    )
    runner.score_responses(
        [
            {
                "id": "retry",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
            }
        ],
        output_path=output,
        judge=object(),
    )
    saved = runner.read_jsonl(output)
    assert [row["id"] for row in saved] == ["retry", "blocked"]
    assert saved[0]["status"] == "ok"


def test_score_retry_interruption_before_replacement_leaves_unique_retry_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    output = tmp_path / "scores.jsonl"
    runner.append_jsonl(
        output, [{"id": "retry", "status": "error", "error": "HTTP 503"}]
    )

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted during replacement")

    monkeypatch.setattr(runner, "_rewrite_jsonl", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.score_responses(
            [
                {
                    "id": "retry",
                    "benchmark": "mmlu_pro",
                    "gold": "A",
                    "extracted_answer": "A",
                }
            ],
            output_path=output,
            judge=object(),
        )
    saved = runner.read_jsonl(output)
    assert [row["id"] for row in saved] == ["retry"]


def test_score_persists_first_result_before_second_judge_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    output = tmp_path / "scores.jsonl"
    calls = 0

    class InterruptingJudge:
        def judge_safety(self, request: str, response: str) -> bool:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("worker interrupted")
            return True

    responses = [
        {
            "id": "first",
            "benchmark": "harmbench",
            "prompt": "p1",
            "raw_response": {"text": "r1"},
            "metadata": {"behavior": "b1"},
        },
        {
            "id": "second",
            "benchmark": "harmbench",
            "prompt": "p2",
            "raw_response": {"text": "r2"},
            "metadata": {"behavior": "b2"},
        },
    ]
    with pytest.raises(KeyboardInterrupt):
        runner.score_responses(responses, output_path=output, judge=InterruptingJudge())

    assert [row["id"] for row in runner.read_jsonl(output)] == ["first"]


def test_score_phase_rejects_orphan_score_artifact_without_manifest(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "results"
    score_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=output)[
        "scores"
    ]
    runner.append_jsonl(score_path, [{"id": "orphan", "status": "ok"}])

    with pytest.raises(RuntimeError, match="missing scoring manifest"):
        runner.score_phase(
            model_id=MODEL_IDS[0],
            response_root=tmp_path / "checkpoints",
            output_root=output,
            judge_factory=lambda: object(),
        )


@pytest.mark.parametrize(
    "message",
    [
        "provider request failed",
        "authentication credentials expired",
        "transport connection reset",
    ],
)
def test_provider_runtime_failures_remain_retryable(
    message: str, tmp_path: Path
) -> None:
    runner = entrypoint()
    output = tmp_path / f"{message.replace(' ', '-')}.jsonl"

    class FailingJudge:
        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise RuntimeError(message)

    rows = runner.score_responses(
        [
            {
                "id": "math",
                "benchmark": "math500",
                "raw_response": {"text": "answer"},
                "gold": "42",
            }
        ],
        output_path=output,
        judge=FailingJudge(),
    )
    assert rows[0]["status"] == "error"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_real_gemini_http_failures_persist_as_retryable_errors(
    status: int, tmp_path: Path
) -> None:
    runner = entrypoint()
    output = tmp_path / f"http-{status}.jsonl"

    class FailingJudge:
        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise JudgeRetryableError(
                f"Gemini request failed with HTTP status {status}"
            )

    rows = runner.score_responses(
        [
            {
                "id": "math",
                "benchmark": "math500",
                "raw_response": {"text": "answer"},
                "gold": "42",
            }
        ],
        output_path=output,
        judge=FailingJudge(),
    )

    assert rows[0]["status"] == "error"
    assert rows[0]["retryable"] is True
    assert rows[0]["error"] == f"Gemini request failed with HTTP status {status}"


def test_missing_candidate_persists_as_retryable_error(tmp_path: Path) -> None:
    runner = entrypoint()
    output = tmp_path / "missing-candidate.jsonl"

    class FailingJudge:
        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise JudgeRetryableError("Gemini response did not contain candidate text")

    rows = runner.score_responses(
        [
            {
                "id": "math",
                "benchmark": "math500",
                "raw_response": {"text": "answer"},
                "gold": "42",
            }
        ],
        output_path=output,
        judge=FailingJudge(),
    )

    assert rows[0]["status"] == "error"
    assert rows[0]["retryable"] is True


def test_parse_failure_persists_as_terminal_unparseable(tmp_path: Path) -> None:
    runner = entrypoint()
    output = tmp_path / "parse.jsonl"

    class FailingJudge:
        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise JudgeParseError(
                "Gemini math judge returned an unparseable JSON result"
            )

    rows = runner.score_responses(
        [
            {
                "id": "math",
                "benchmark": "math500",
                "raw_response": {"text": "answer"},
                "gold": "42",
            }
        ],
        output_path=output,
        judge=FailingJudge(),
    )

    assert rows[0]["status"] == "unparseable"
    assert rows[0]["retryable"] is False


def test_score_retry_restoration_uses_membership_set_and_preserves_order(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "ordered.jsonl"
    runner.append_jsonl(
        output,
        [
            {"id": "keep", "status": "ok"},
            {"id": "retry-a", "status": "error", "error": "HTTP 503"},
            {"id": "retry-b", "status": "error", "error": "HTTP 503"},
        ],
    )
    rows = runner.score_responses(
        [
            {
                "id": "retry-a",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
            },
            {
                "id": "retry-b",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "B",
            },
        ],
        output_path=output,
        judge=object(),
    )

    assert [row["id"] for row in rows] == ["retry-a", "retry-b"]
    assert [row["id"] for row in runner.read_jsonl(output)] == [
        "keep",
        "retry-a",
        "retry-b",
    ]


def test_non_ok_retry_response_replaces_existing_row_without_duplicate(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "non-ok-retry.jsonl"
    runner.append_jsonl(
        output,
        [{"id": "x", "status": "error", "error": "HTTP 503", "retryable": True}],
    )
    response = {
        "id": "x",
        "benchmark": "math500",
        "status": "error",
        "error": "generation failed",
        "gold": "42",
    }

    class ExplodingJudge:
        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise AssertionError("non-OK response must not call judge")

    written = runner.score_responses(
        [response], output_path=output, judge=ExplodingJudge()
    )
    saved = runner.read_jsonl(output)
    assert [row["id"] for row in saved] == ["x"]
    assert saved[0]["status"] == "error"
    assert saved[0]["error"] == "generation failed"
    assert written == saved

    assert (
        runner.score_responses([response], output_path=output, judge=ExplodingJudge())
        == []
    )
    assert runner.read_jsonl(output) == saved


def test_mixed_incomplete_ppl_defers_complete_invalid_validation(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [
            {
                "id": "complete-invalid",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
                "generated_token_count": 1,
                "selected_generated_token_logprobs": [None],
            },
            {
                "id": "global-error",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "error",
                "generated_token_count": 0,
                "selected_generated_token_logprobs": [],
            },
        ],
    )

    summary = runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: (_ for _ in ()).throw(AssertionError("judge called")),
    )

    assert summary["ppl"] == {
        "ppl": None,
        "selected_token_count": 1,
        "generated_token_count": 1,
        "covered_records": 1,
        "total_records": 2,
        "coverage_ratio": 1.0,
    }
    score_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=output)[
        "scores"
    ]
    assert [row["id"] for row in runner.read_jsonl(score_path)] == [
        "complete-invalid",
        "global-error",
    ]


def test_interleaved_retry_rows_replace_in_place_and_resume_after_interruptions(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    output = tmp_path / "scores.jsonl"
    original = [
        {"id": "keep-a", "status": "ok"},
        {"id": "retry-a", "status": "error", "error": "503"},
        {"id": "keep-b", "status": "blocked"},
        {"id": "retry-b", "status": "error", "error": "503"},
        {"id": "terminal", "status": "unparseable"},
    ]
    runner.append_jsonl(output, original)
    responses = [
        {
            "id": "retry-a",
            "benchmark": "harmbench",
            "raw_response": {"text": "answer-a"},
            "metadata": {"behavior": "behavior-a"},
        },
        {
            "id": "retry-b",
            "benchmark": "harmbench",
            "raw_response": {"text": "answer-b"},
            "metadata": {"behavior": "behavior-b"},
        },
    ]

    class InterruptBeforeJudge:
        def judge_safety(self, request: str, response: str) -> bool:
            raise KeyboardInterrupt("before replacement")

        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise AssertionError("unexpected math judge")

    with pytest.raises(KeyboardInterrupt):
        runner.score_responses(
            responses, output_path=output, judge=InterruptBeforeJudge()
        )
    assert runner.read_jsonl(output) == original

    calls = 0

    class InterruptAfterFirst:
        def judge_safety(self, request: str, response: str) -> bool:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("after first replacement")
            return True

        def judge_math(self, response: str, expected: str) -> dict[str, bool]:
            raise AssertionError("unexpected math judge")

    with pytest.raises(KeyboardInterrupt):
        runner.score_responses(
            responses, output_path=output, judge=InterruptAfterFirst()
        )
    interrupted = runner.read_jsonl(output)
    assert [row["id"] for row in interrupted] == [
        "keep-a",
        "retry-a",
        "keep-b",
        "retry-b",
        "terminal",
    ]
    assert interrupted[1]["status"] == "ok"
    assert interrupted[3]["status"] == "error"

    class ResumeJudge:
        def judge_safety(self, request: str, response: str) -> bool:
            return True

    close_output = tmp_path / "close-scores.jsonl"
    runner.append_jsonl(close_output, original)
    stream = runner._score_responses_iterable(
        iter(responses), output_path=close_output, judge=ResumeJudge()
    )
    next(stream)
    stream.close()
    closed = runner.read_jsonl(close_output)
    assert [row["id"] for row in closed] == [
        "keep-a",
        "retry-a",
        "keep-b",
        "retry-b",
        "terminal",
    ]
    assert closed[1]["status"] == "ok"
    assert closed[3]["status"] == "error"

    written = runner.score_responses(responses, output_path=output, judge=ResumeJudge())
    assert [row["id"] for row in written] == ["retry-b"]
    final = runner.read_jsonl(output)
    assert [row["id"] for row in final] == [
        "keep-a",
        "retry-a",
        "keep-b",
        "retry-b",
        "terminal",
    ]
    assert len({str(row["id"]) for row in final}) == len(final)
    assert final[1]["status"] == "ok"
    assert final[3]["status"] == "ok"


def test_generation_summary_does_not_report_unscored_rows_as_zero_accuracy() -> None:
    runner = entrypoint()
    summary = runner.aggregate_records(
        [{"id": "m", "benchmark": "mmlu_pro", "status": "ok"}],
        model_id=MODEL_IDS[0],
    )
    assert summary["scores"]["mmlu_pro"]["accuracy"] is None
    assert summary["scores"]["mmlu_pro"]["status"] == "pending"


def test_score_phase_uses_only_canonical_slug_aggregate_root(tmp_path: Path) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [
            {
                "id": "m",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
            }
        ],
    )
    runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: object(),
    )
    assert (output / runner.model_spec(MODEL_IDS[0]).slug / "summary.json").exists()
    assert not (output / MODEL_IDS[0] / "summary.json").exists()


def test_generation_knobs_are_validated_and_engine_kwargs_are_bounded() -> None:
    runner = entrypoint()
    assert runner.generation_settings(256, 4096, 0.8) == {
        "max_tokens": 256,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "temperature": 0.0,
        "logprobs": 1,
    }
    with pytest.raises(ValueError, match="max_tokens"):
        runner.generation_settings(0, 4096, 0.8)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        runner.generation_settings(256, 4096, 1.1)


def test_score_phase_reads_checkpoint_responses_but_writes_formal_results_to_output(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [
            {
                "id": "0",
                "benchmark": "mmlu_pro",
                "prompt": "q",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
                "metadata": {},
            }
        ],
    )
    summary = runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: object(),
    )
    score_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=output)[
        "scores"
    ]
    assert (
        json.loads(
            (output / runner.model_spec(MODEL_IDS[0]).slug / "summary.json").read_text()
        )
        == summary
    )
    assert len(runner.read_jsonl(score_path)) == 1
    assert not no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "scores"
    ].exists()


def test_scoring_manifest_hashes_in_bounded_chunks_and_preserves_ordered_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    existing = tmp_path / "responses.jsonl"
    payload = b"x" * (2 * 1024 * 1024 + 17)
    existing.write_bytes(payload)
    missing = tmp_path / "missing.jsonl"
    paths = {"harmbench": existing, "mmlu_pro": missing}

    class BoundedReader:
        def __init__(self, handle: Any) -> None:
            self.handle = handle
            self.sizes: list[int] = []

        def __enter__(self) -> "BoundedReader":
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            self.sizes.append(size)
            return self.handle.read(size)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

    original_open = Path.open
    readers: list[BoundedReader] = []

    def open_with_spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(self, *args, **kwargs)
        if "b" in str(kwargs.get("mode", args[0] if args else "r")):
            reader = BoundedReader(handle)
            readers.append(reader)
            return reader
        return handle

    monkeypatch.setattr(Path, "open", open_with_spy)
    manifest = runner._scoring_manifest(
        model_id=MODEL_IDS[0],
        response_root=tmp_path,
        response_paths=paths,
        response_ids={"harmbench": ["h0", "h1"], "mmlu_pro": []},
        judge_model=runner.DEFAULT_JUDGE_MODEL,
    )
    assert manifest["response_files"]["harmbench"]["ids"] == ["h0", "h1"]
    assert manifest["response_files"]["mmlu_pro"]["ids"] == []
    assert (
        manifest["response_files"]["mmlu_pro"]["content_sha256"]
        == hashlib.sha256(b"").hexdigest()
    )
    assert readers and all(
        0 < size <= runner.HASH_CHUNK_SIZE
        for reader in readers
        for size in reader.sizes
    )


def test_score_phase_preflights_duplicate_response_ids_before_manifest_or_judge(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths(
        "harmbench", MODEL_IDS[0], root=checkpoint
    )["responses"]
    runner.append_jsonl(
        response_path,
        [
            {"id": "duplicate", "benchmark": "harmbench", "status": "ok"},
            {"id": "duplicate", "benchmark": "harmbench", "status": "ok"},
        ],
    )
    score_path = no_steering.output_paths("harmbench", MODEL_IDS[0], root=output)[
        "scores"
    ]
    calls: list[str] = []
    with pytest.raises(ValueError, match="duplicate"):
        runner.score_phase(
            model_id=MODEL_IDS[0],
            response_root=checkpoint,
            output_root=output,
            judge_factory=lambda: calls.append("constructed"),
        )
    assert calls == []
    assert not (
        output / runner.model_spec(MODEL_IDS[0]).slug / "scoring_manifest.json"
    ).exists()
    assert not score_path.exists()


def test_score_phase_streams_response_rows_without_materializing_response_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [{"id": "m0", "benchmark": "mmlu_pro", "gold": "A", "extracted_answer": "A"}],
    )
    original_read_jsonl = runner.read_jsonl

    def guarded_read_jsonl(path: str | Path) -> list[dict[str, object]]:
        if Path(path).resolve().is_relative_to(checkpoint.resolve()):
            raise AssertionError("response file was materialized")
        return original_read_jsonl(path)

    monkeypatch.setattr(runner, "read_jsonl", guarded_read_jsonl)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("read_bytes used")),
    )
    summary = runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: object(),
    )
    assert summary["ppl"]["total_records"] == 1
    assert summary["scores"]["mmlu_pro"]["correct"] == 1


@pytest.mark.parametrize("benchmark", ["harmbench", "math500"])
def test_score_phase_does_not_construct_judge_for_existing_empty_response_file(
    tmp_path: Path, benchmark: str
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths(benchmark, MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.touch()
    calls: list[str] = []

    runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: calls.append("constructed"),
    )

    assert calls == []


def test_score_phase_incomplete_invalid_ppl_matches_ppl_summary_without_raising(
    tmp_path: Path,
) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [
            {
                "id": "partial-invalid",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
                "generated_token_count": 2,
                "selected_generated_token_logprobs": [None],
            }
        ],
    )

    summary = runner.score_phase(
        model_id=MODEL_IDS[0],
        response_root=checkpoint,
        output_root=output,
        judge_factory=lambda: (_ for _ in ()).throw(AssertionError("judge called")),
    )

    assert summary["ppl"] == {
        "ppl": None,
        "selected_token_count": 1,
        "generated_token_count": 2,
        "covered_records": 1,
        "total_records": 1,
        "coverage_ratio": 0.5,
    }


def test_score_phase_complete_invalid_ppl_still_raises(tmp_path: Path) -> None:
    runner = entrypoint()
    checkpoint = tmp_path / "checkpoints"
    output = tmp_path / "results"
    response_path = no_steering.output_paths("mmlu_pro", MODEL_IDS[0], root=checkpoint)[
        "responses"
    ]
    runner.append_jsonl(
        response_path,
        [
            {
                "id": "complete-invalid",
                "benchmark": "mmlu_pro",
                "gold": "A",
                "extracted_answer": "A",
                "status": "ok",
                "generated_token_count": 1,
                "selected_generated_token_logprobs": [None],
            }
        ],
    )

    with pytest.raises(ValueError, match="logprobs must be finite real numbers"):
        runner.score_phase(
            model_id=MODEL_IDS[0],
            response_root=checkpoint,
            output_root=output,
            judge_factory=lambda: (_ for _ in ()).throw(AssertionError("judge called")),
        )


def test_main_keeps_tee_and_disables_child_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint()
    calls: list[tuple[str, object]] = []

    class FakeContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_notify(*args: object, **kwargs: object) -> FakeContext:
        calls.append(("notify", kwargs))
        return FakeContext()

    def fake_tee(*args: object, **kwargs: object) -> FakeContext:
        calls.append(("tee", kwargs))
        return FakeContext()

    monkeypatch.setattr(runner, "notify_on_exit", fake_notify)
    monkeypatch.setattr(runner, "tee_stdout", fake_tee)
    monkeypatch.setattr(runner, "_run", lambda args: None)

    runner.main(["--model-id", MODEL_IDS[0], "--phase", "preflight"])

    assert [name for name, _ in calls] == ["notify", "tee"]
    assert calls[0][1] == {
        "log_file": runner.ROOT / "logs" / "no_steering.log",
        "enabled": False,
    }
