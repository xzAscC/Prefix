from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prefix.data import load_harmbench, load_llm_lat
from prefix.judge import GeminiJudge, JudgeBlocked, judge_batch
from prefix.metrics import binned_success_rate, mean_trajectories_by_label
from prefix.notify import notify_on_exit
from prefix.runner import (
    CaptureSink,
    DirectionRecord,
    append_jsonl,
    build_dim_directions,
    chat_prompt,
    completed_ids,
    get_engine,
    load_config,
    load_directions,
    read_jsonl,
    require_complete,
    save_directions,
    steered_generate,
    tee_stdout,
    verify_manifest,
    write_json_atomic,
    write_manifest,
)

CHECKPOINTS = Path("checkpoints")
FIGS = Path("figs")


def apply_limit(config: dict[str, Any], limit: int | None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if limit is not None and limit <= 8:
        result["direction"]["n_per_class"] = limit
    return result


def trajectory_records(
    sink: CaptureSink, request_to_id: dict[str, str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sink.rows:
        if row.get("phase") == "decode" and str(row.get("request_id")) in request_to_id:
            grouped[str(row["request_id"])].append(row)
    records = []
    for request_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["k"]))
        records.append(
            {
                "id": request_to_id[request_id],
                "c": [float(row["dots"][0]) / float(row["norm"]) for row in rows],
            }
        )
    return sorted(records, key=lambda row: int(row["id"]))


def direction_phase(
    config: dict[str, Any], config_path: str | Path
) -> dict[int, DirectionRecord]:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    direction_path = CHECKPOINTS / "exp1_direction.json"
    manifest_path = CHECKPOINTS / "exp1_manifest.json"
    if direction_path.exists():
        verify_manifest(manifest_path, config)
        print("direction: checkpoint exists; resuming")
        return load_directions(direction_path)
    model = config["model"]
    llm = get_engine(
        model["id"],
        max_model_len=model.get("max_model_len"),
        gpu_memory_utilization=float(model.get("gpu_memory_utilization", 0.9)),
    )
    direction = config["direction"]
    n = int(direction["n_per_class"])
    pos = load_llm_lat(direction["pos_dataset"], n)
    neg = load_llm_lat(direction["neg_dataset"], n)
    records = build_dim_directions(llm, pos, neg, [int(model["layer"])])
    save_directions(direction_path, records)
    write_manifest(manifest_path, config)
    print("direction: saved")
    return records


def _tokenizer(llm: Any) -> Any:
    return llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else None


def generate_harmbench(
    llm: Any,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    direction: DirectionRecord,
    generation_path: Path,
    trajectory_path: Path,
    batch_prompts: int,
) -> None:
    expected = [str(index) for index in range(len(records))]
    tokenizer = _tokenizer(llm)
    layer = int(config["model"]["layer"])
    pending = sorted(set(expected) - completed_ids(generation_path))
    while pending:
        chunk = pending[:batch_prompts]
        prompts = [
            chat_prompt(tokenizer, str(records[int(index)]["behavior"]), False)
            for index in chunk
        ]
        sink = CaptureSink()
        outputs = steered_generate(
            llm,
            prompts,
            max_tokens=int(config["generation"]["max_new_tokens"]),
            spec=None,
            batch_prompts=batch_prompts,
            sink=sink,
            scalar_directions=[direction.direction],
            capture_layer=layer,
        )
        request_to_id = {
            str(output.request_id): index for index, output in zip(chunk, outputs)
        }
        append_jsonl(
            generation_path,
            [
                {"id": index, "text": output.text}
                for index, output in zip(chunk, outputs)
            ],
        )
        append_jsonl(trajectory_path, trajectory_records(sink, request_to_id))
        print(f"generate: completed {len(chunk)} records", flush=True)
        pending = sorted(set(expected) - completed_ids(generation_path))


def judge_harmbench(
    generation_path: Path,
    judge_path: Path,
    records: list[dict[str, Any]],
    judge: Any,
) -> None:
    expected = [str(index) for index in range(len(records))]
    require_complete(generation_path, expected, label="generations")
    generations = {
        str(row["id"]): str(row["text"]) for row in read_jsonl(generation_path)
    }
    pending = sorted(set(expected) - completed_ids(judge_path))
    while pending:

        def one(index: str) -> tuple[bool | None, bool]:
            try:
                return (
                    bool(
                        judge.judge_safety(
                            str(records[int(index)]["behavior"]), generations[index]
                        )
                    ),
                    False,
                )
            except JudgeBlocked:
                return None, True

        values = judge_batch(one, pending)
        append_jsonl(
            judge_path,
            [
                {"id": index, "safe": safe, "blocked": blocked}
                for index, (safe, blocked) in zip(pending, values)
            ],
        )
        print(f"judge: completed {len(pending)} records", flush=True)
        pending = sorted(set(expected) - completed_ids(judge_path))


def early_bins(
    values: np.ndarray, labels: np.ndarray, n_bins: int
) -> list[dict[str, Any]]:
    return [
        {"bin_center": item.bin_center, "rate": item.rate, "count": item.count}
        for item in binned_success_rate(values, labels, n_bins)
    ]


def analyze(
    config: dict[str, Any],
    judge_path: Path,
    trajectory_path: Path,
    expected: list[str],
    results_path: Path,
) -> dict[str, Any]:
    judge_data = read_jsonl(judge_path)
    blocked = sum(1 for row in judge_data if row.get("blocked", False))
    if blocked:
        print(
            f"analyze: warning, excluding {blocked} blocked judge records", flush=True
        )
    judge_rows = {
        str(row["id"]): bool(row["safe"])
        for row in judge_data
        if not row.get("blocked", False)
    }
    trace_rows = {
        str(row["id"]): np.asarray(row["c"], dtype=float)
        for row in read_jsonl(trajectory_path)
    }
    ids = [index for index in expected if index in judge_rows and index in trace_rows]
    missing = [index for index in expected if index not in trace_rows]
    if missing:
        print(
            f"analyze: warning, skipping {len(missing)} ids without traces", flush=True
        )
    traces = [trace_rows[index] for index in ids]
    labels = np.asarray([judge_rows[index] for index in ids], dtype=bool)
    means = mean_trajectories_by_label(traces, labels)
    early: dict[str, list[dict[str, Any]]] = {}
    for token in config["analysis"]["early_tokens"]:
        available = [
            (trace[token - 1], judge_rows[index])
            for index, trace in zip(ids, traces)
            if len(trace) >= token
        ]
        if available:
            values, values_labels = zip(*available)
            early[str(token)] = early_bins(
                np.asarray(values),
                np.asarray(values_labels),
                int(config["analysis"]["n_bins"]),
            )
        else:
            early[str(token)] = []
    result = {
        "mean_ct_by_label": {
            str(label).lower(): values.tolist() for label, values in means.items()
        },
        "gt_early": early,
        "n": len(ids),
        "blocked": blocked,
    }
    write_json_atomic(results_path, result)
    FIGS.mkdir(parents=True, exist_ok=True)
    for label, values in means.items():
        plt.plot(np.arange(len(values)), values, label="safe" if label else "unsafe")
    plt.xlabel("token index")
    plt.ylabel("mean cosine alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "exp1_ct_by_label.pdf")
    plt.close()
    for token, bins in early.items():
        plt.plot(
            [item["bin_center"] for item in bins],
            [item["rate"] for item in bins],
            marker="o",
            label=f"t={token}",
        )
    plt.xlabel("cosine alignment")
    plt.ylabel("safe rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "exp1_gt_early.pdf")
    plt.close()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp1.yaml")
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
    with notify_on_exit("exp1", log_file="logs/exp1.log"):
        with tee_stdout("logs/exp1.log"):
            direction = None
            if "direction" in phases:
                direction = direction_phase(config, args.config)[
                    int(config["model"]["layer"])
                ]
            records = (
                load_harmbench()[: args.limit]
                if args.limit is not None
                else load_harmbench()
            )
            expected = [str(index) for index in range(len(records))]
            generation = CHECKPOINTS / "exp1_generations.jsonl"
            trajectories = CHECKPOINTS / "exp1_trajectories.jsonl"
            if "generate" in phases:
                if direction is None:
                    direction = load_directions(CHECKPOINTS / "exp1_direction.json")[
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
                    direction,
                    generation,
                    trajectories,
                    args.batch_prompts,
                )
            if "judge" in phases:
                judge = GeminiJudge(
                    model=config["judge"]["model"], region=config["judge"]["region"]
                )
                judge_harmbench(
                    generation, CHECKPOINTS / "exp1_judge.jsonl", records, judge
                )
            if "analyze" in phases:
                require_complete(
                    CHECKPOINTS / "exp1_judge.jsonl", expected, label="judge"
                )
                analyze(
                    config,
                    CHECKPOINTS / "exp1_judge.jsonl",
                    trajectories,
                    expected,
                    CHECKPOINTS / "exp1_results.json",
                )


if __name__ == "__main__":
    main()
