from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prefix.data import load_harmbench, load_llm_lat, load_mmlu_pro, mmlu_exp2_subset
from prefix.judge import GeminiJudge, JudgeBlocked, judge_batch
from prefix.notify import notify_on_exit
from prefix.runner import (
    CaptureSink,
    DirectionRecord,
    SteeringSpec,
    append_jsonl,
    build_dim_directions,
    chat_prompt,
    completed_ids,
    condition_id,
    get_engine,
    load_config,
    load_directions,
    mmlu_prompt,
    parse_answer_letter,
    read_jsonl,
    require_complete,
    save_directions,
    steered_generate,
    tee_stdout,
    verify_manifest,
    write_json_atomic,
    write_manifest,
)
from prefix.steering import SteeringSchedule

CHECKPOINTS = Path("checkpoints")
FIGS = Path("figs")


def apply_limit(config: dict[str, Any], limit: int | None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if limit is not None and limit <= 8:
        result["steering"]["alphas"] = list(result["steering"]["alphas"])[:3]
        if "mmlu" in result:
            result["mmlu"]["n"] = min(int(result["mmlu"]["n"]), limit)
    return result


def direction_phase(config: dict[str, Any]) -> dict[int, DirectionRecord]:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    direction_path = CHECKPOINTS / "exp2_direction.json"
    manifest_path = CHECKPOINTS / "exp2_manifest.json"
    if direction_path.exists():
        verify_manifest(manifest_path, config)
        return load_directions(direction_path)
    model = config["model"]
    llm = get_engine(
        model["id"],
        max_model_len=model.get("max_model_len"),
        gpu_memory_utilization=float(model.get("gpu_memory_utilization", 0.9)),
    )
    direction = (
        config["direction"]
        if "direction" in config
        else {
            "pos_dataset": "LLM-LAT/benign-dataset",
            "neg_dataset": "LLM-LAT/harmful-dataset",
            "n_per_class": 100,
        }
    )
    pos = load_llm_lat(str(direction["pos_dataset"]), int(direction["n_per_class"]))
    neg = load_llm_lat(str(direction["neg_dataset"]), int(direction["n_per_class"]))
    records = build_dim_directions(llm, pos, neg, [int(model["layer"])])
    save_directions(direction_path, records)
    write_manifest(manifest_path, config)
    return records


def _tokenizer(llm: Any) -> Any:
    return llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else None


def _condition_file(checkpoint_dir: Path, prefix: str, cond: str) -> Path:
    return checkpoint_dir / f"exp2_{prefix}_{cond.replace('/', '_')}.jsonl"


def _spec(alpha: float, direction: DirectionRecord, layer: int) -> SteeringSpec | None:
    if alpha == 0.0:
        return None
    return SteeringSpec(
        layer, direction.direction, alpha, direction.mean_norm, SteeringSchedule.full()
    )


def generate_harmbench(
    llm: Any,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    alphas: list[float],
    direction: DirectionRecord,
    checkpoint_dir: Path,
    batch_prompts: int,
) -> None:
    tokenizer = _tokenizer(llm)
    layer = int(config["model"]["layer"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected = [str(index) for index in range(len(records))]
    for alpha in alphas:
        cond = condition_id("exp2", "hb", float(alpha))
        path = _condition_file(checkpoint_dir, "hb", cond)
        pending = sorted(set(expected) - completed_ids(path))
        while pending:
            chunk = pending[:batch_prompts]
            prompts = [
                chat_prompt(tokenizer, str(records[int(index)]["behavior"]), False)
                for index in chunk
            ]
            outputs = steered_generate(
                llm,
                prompts,
                max_tokens=int(config["harmbench"]["max_new_tokens"]),
                spec=_spec(float(alpha), direction, layer),
                batch_prompts=batch_prompts,
            )
            append_jsonl(
                path,
                [
                    {"id": index, "text": output.text}
                    for index, output in zip(chunk, outputs)
                ],
            )
            print(
                f"generate hb alpha={alpha}: completed {len(chunk)} records", flush=True
            )
            pending = sorted(set(expected) - completed_ids(path))


def generate_mmlu(
    llm: Any,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    alphas: list[float],
    direction: DirectionRecord,
    checkpoint_dir: Path,
    batch_prompts: int,
) -> None:
    tokenizer = _tokenizer(llm)
    layer = int(config["model"]["layer"])
    indices = mmlu_exp2_subset(
        int(config["mmlu"]["n"]), int(config["seed"]), n_total=len(records)
    )
    expected = [str(index) for index in indices]
    for alpha in alphas:
        cond = condition_id("exp2", "mmlu", float(alpha))
        path = _condition_file(checkpoint_dir, "mmlu", cond)
        pending = sorted(set(expected) - completed_ids(path))
        while pending:
            chunk = pending[:batch_prompts]
            prompts = [
                chat_prompt(
                    tokenizer,
                    mmlu_prompt(
                        str(records[int(index)]["question"]),
                        list(records[int(index)]["options"]),
                    ),
                    True,
                )
                for index in chunk
            ]
            outputs = steered_generate(
                llm,
                prompts,
                int(config["mmlu"]["max_new_tokens"]),
                _spec(float(alpha), direction, layer),
                batch_prompts=batch_prompts,
            )
            append_jsonl(
                path,
                [
                    {
                        "id": index,
                        "text": output.text,
                        "pred": parse_answer_letter(output.text),
                    }
                    for index, output in zip(chunk, outputs)
                ],
            )
            print(
                f"generate mmlu alpha={alpha}: completed {len(chunk)} records",
                flush=True,
            )
            pending = sorted(set(expected) - completed_ids(path))


def judge_harmbench(
    generation_dir: Path,
    judge_path: Path,
    records: list[dict[str, Any]],
    alphas: list[float],
    judge: Any,
) -> None:
    expected = [str(index) for index in range(len(records))]
    jobs: list[tuple[str, float, str, str]] = []
    for alpha in alphas:
        cond = condition_id("exp2", "hb", float(alpha))
        path = _condition_file(generation_dir, "hb", cond)
        require_complete(path, expected, label=f"generations {cond}")
        for row in read_jsonl(path):
            jobs.append(
                (f"{cond}/{row['id']}", float(alpha), str(row["id"]), str(row["text"]))
            )
    pending = [job for job in jobs if job[0] not in completed_ids(judge_path)]
    if pending:

        def one(job: tuple[str, float, str, str]) -> tuple[bool | None, bool]:
            try:
                return (
                    bool(
                        judge.judge_safety(
                            str(records[int(job[2])]["behavior"]), job[3]
                        )
                    ),
                    False,
                )
            except JudgeBlocked:
                return None, True
            except RuntimeError:
                return None, False

        for start in range(0, len(pending), 64):
            chunk = pending[start : start + 64]
            values = judge_batch(one, chunk)
            append_jsonl(
                judge_path,
                [
                    {
                        "id": job[0],
                        "alpha": job[1],
                        "i": int(job[2]),
                        "safe": safe,
                        "blocked": blocked,
                    }
                    for job, (safe, blocked) in zip(chunk, values)
                ],
            )
    require_complete(
        judge_path,
        [
            f"{condition_id('exp2', 'hb', float(alpha))}/{index}"
            for alpha in alphas
            for index in expected
        ],
        label="judge",
    )
    print(f"judge: completed {len(pending)} records", flush=True)


def analyze(
    config: dict[str, Any],
    generation_dir: Path,
    judge_path: Path,
    mmlu_records: list[dict[str, Any]],
    alphas: list[float],
    results_path: Path,
) -> dict[str, Any]:
    judge_rows = read_jsonl(judge_path)
    p_safe: dict[str, float] = {}
    blocked: dict[str, int] = {}
    for alpha in alphas:
        values = [
            bool(row["safe"])
            for row in judge_rows
            if float(row["alpha"]) == float(alpha) and row["safe"] is not None
        ]
        blocked[str(alpha)] = sum(
            1
            for row in judge_rows
            if float(row["alpha"]) == float(alpha) and row.get("blocked", False)
        )
        p_safe[str(alpha)] = float(np.mean(values)) if values else float("nan")
    blocked_total = sum(blocked.values())
    if blocked_total:
        print(
            f"analyze: warning, excluding {blocked_total} blocked judge records",
            flush=True,
        )
    indices = mmlu_exp2_subset(
        int(config["mmlu"]["n"]), int(config["seed"]), n_total=len(mmlu_records)
    )
    expected = [str(index) for index in indices]
    a_mmlu: dict[str, float] = {}
    for alpha in alphas:
        cond = condition_id("exp2", "mmlu", float(alpha))
        rows = read_jsonl(_condition_file(generation_dir, "mmlu", cond))
        require_complete(
            _condition_file(generation_dir, "mmlu", cond),
            expected,
            label=f"mmlu {cond}",
        )
        correct = [
            row.get("pred") == mmlu_records[int(row["id"])]["answer_letter"]
            for row in rows
        ]
        a_mmlu[str(alpha)] = float(np.mean(correct)) if correct else float("nan")
    result = {"alpha": alphas, "p_safe": p_safe, "a_mmlu": a_mmlu, "blocked": blocked}
    write_json_atomic(results_path, result)
    FIGS.mkdir(parents=True, exist_ok=True)
    x = np.asarray(alphas, dtype=float)
    plot_x = np.where(x == 0, np.finfo(float).tiny, x)
    plt.plot(
        plot_x, [p_safe[str(alpha)] for alpha in alphas], marker="o", label="P_safe"
    )
    plt.plot(
        plot_x, [a_mmlu[str(alpha)] for alpha in alphas], marker="o", label="A_MMLU"
    )
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "exp2_tradeoff.pdf")
    plt.close()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp2.yaml")
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["direction", "generate", "judge", "analyze", "all"],
        default=["all"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-prompts", type=int, default=256)
    args = parser.parse_args(argv)
    config = apply_limit(load_config(args.config), args.limit)
    phases = (
        ["direction", "generate", "judge", "analyze"]
        if "all" in args.phase
        else args.phase
    )
    with notify_on_exit("exp2", log_file="logs/exp2.log"):
        with tee_stdout("logs/exp2.log"):
            direction = (
                direction_phase(config)[int(config["model"]["layer"])]
                if "direction" in phases
                else None
            )
            alphas = [float(value) for value in config["steering"]["alphas"]]
            records = (
                load_harmbench()[: args.limit]
                if args.limit is not None
                else load_harmbench()
            )
            if "generate" in phases:
                if direction is None:
                    direction = load_directions(CHECKPOINTS / "exp2_direction.json")[
                        int(config["model"]["layer"])
                    ]
                llm = get_engine(
                    config["model"]["id"],
                    max_model_len=config["model"].get("max_model_len"),
                    gpu_memory_utilization=float(
                        config["model"].get("gpu_memory_utilization", 0.9)
                    ),
                )
                generate_harmbench(
                    llm,
                    config,
                    records,
                    alphas,
                    direction,
                    CHECKPOINTS,
                    args.batch_prompts,
                )
                mmlu_records = load_mmlu_pro(config["mmlu"]["source"])
                if args.limit is not None:
                    mmlu_records = mmlu_records[: args.limit]
                generate_mmlu(
                    llm,
                    config,
                    mmlu_records,
                    alphas,
                    direction,
                    CHECKPOINTS,
                    args.batch_prompts,
                )
            if "judge" in phases:
                judge_harmbench(
                    CHECKPOINTS,
                    CHECKPOINTS / "exp2_judge.jsonl",
                    records,
                    alphas,
                    GeminiJudge(
                        model=config["judge"]["model"], region=config["judge"]["region"]
                    ),
                )
            if "analyze" in phases:
                mmlu_records = load_mmlu_pro(config["mmlu"]["source"])
                if args.limit is not None:
                    mmlu_records = mmlu_records[: args.limit]
                analyze(
                    config,
                    CHECKPOINTS,
                    CHECKPOINTS / "exp2_judge.jsonl",
                    mmlu_records,
                    alphas,
                    CHECKPOINTS / "exp2_results.json",
                )


if __name__ == "__main__":
    main()
