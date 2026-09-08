from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from prefix.data import load_harmbench, load_llm_lat, load_mmlu_pro, harmbench_split
from prefix.judge import GeminiJudge, JudgeBlocked, judge_batch
from prefix.notify import notify_on_exit
from prefix.metrics import pareto_frontier, select_operating_point
from prefix.runner import (
    SteeringSpec,
    append_jsonl,
    build_dim_directions,
    chat_prompt,
    completed_ids,
    get_engine,
    load_config,
    load_directions,
    mmlu_prompt,
    parse_answer_letter,
    read_json,
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
from prefix.vllm_steering import attach_steering, make_decode_index_resolver

SamplingParams = None


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    schedule: str
    layer: int
    alpha: float

    @property
    def name(self) -> str:
        return f"{self.schedule}_l{self.layer}_a{repr(self.alpha)}"


def schedule_for(name: str) -> SteeringSchedule:
    if name == "full":
        return SteeringSchedule.full()
    if name == "prefix-5":
        return SteeringSchedule.prefix(5)
    if name == "one-token":
        return SteeringSchedule.one_token()
    raise ValueError(f"unknown schedule: {name}")


def _smoke_layers(cfg: dict[str, Any], limit: int | None) -> list[int]:
    layers = [int(x) for x in cfg["grid"]["layers"]]
    if limit is not None and limit <= 8:
        layers = [layers[len(layers) // 2]]
    return layers


def validation_conditions(
    cfg: dict[str, Any], limit: int | None = None
) -> list[Condition]:
    layers = _smoke_layers(cfg, limit)
    grids = cfg["grid"]["alphas_by_schedule"]
    if limit is not None and limit <= 8:
        grids = {key: list(value)[:2] for key, value in grids.items()}
    return [
        Condition(schedule, int(layer), float(alpha))
        for schedule in cfg["schedules"]
        for layer in layers
        for alpha in grids[schedule]
    ]


def choose_selection(
    points: list[tuple[float, float]], baseline: float, cap: float
) -> tuple[int, bool]:
    frontier = pareto_frontier(points)
    try:
        selected = select_operating_point([points[i] for i in frontier], baseline, cap)
        return frontier[selected], False
    except ValueError:
        return max(frontier, key=lambda i: points[i][1]), True


def require_selection(path: Path) -> dict[str, Any]:
    selection = read_json(path)
    if selection is None:
        raise RuntimeError(
            f"selection is missing at {path}; run judge and analyze first"
        )
    return selection


def generate_jsonl(
    llm: Any,
    prompts: list[str],
    path: Path,
    spec: SteeringSpec | None,
    max_tokens: int,
    batch_prompts: int,
    ids: list[str] | None = None,
) -> None:
    done = completed_ids(path)
    record_ids = ids if ids is not None else [str(i) for i in range(len(prompts))]
    pending = [
        (item_id, prompt)
        for item_id, prompt in zip(record_ids, prompts)
        if item_id not in done
    ]
    for start in range(0, len(pending), batch_prompts):
        chunk = pending[start : start + batch_prompts]
        results = steered_generate(
            llm,
            [prompt for _, prompt in chunk],
            max_tokens=max_tokens,
            spec=spec,
            batch_prompts=batch_prompts,
        )
        append_jsonl(
            path,
            [
                {"id": item_id, "text": result.text, "request_id": result.request_id}
                for (item_id, _), result in zip(chunk, results)
            ],
        )


def judge_jsonl(
    generation: Path,
    output: Path,
    expected: list[str],
    judge: Callable[[dict[str, Any]], Any],
    output_id: Callable[[dict[str, Any]], str] | None = None,
    blocked_result: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    require_complete(generation, expected, label=str(generation))
    done = completed_ids(output)
    rows = read_jsonl(generation)
    pending = [row for row in rows if str(row["id"]) not in done]

    def judge_item(row: dict[str, Any]) -> dict[str, Any]:
        try:
            result = judge(row)
        except JudgeBlocked:
            result = blocked_result(row) if blocked_result else {}
            result["blocked"] = True
            return result
        except RuntimeError:
            return {"safe": None, "blocked": False}
        result = result if isinstance(result, dict) else {"safe": bool(result)}
        result["blocked"] = False
        return result

    results = judge_batch(judge_item, pending, max_workers=8)
    append_jsonl(
        output,
        [
            {"id": output_id(row) if output_id else str(row["id"]), **result}
            for row, result in zip(pending, results)
        ],
    )


def logprob_pass(
    llm: Any,
    tokenizer: Any,
    prompts: list[str],
    path: Path,
    condition: str,
    spec: SteeringSpec | None,
    batch_prompts: int,
) -> None:
    done = {(str(row["condition"]), str(row["id"])) for row in read_jsonl(path)}
    detacher = None
    resolver = make_decode_index_resolver(llm)
    try:
        if spec is not None:
            detacher = attach_steering(
                llm,
                spec.layer,
                spec.direction,
                spec.alpha,
                spec.mean_norm,
                spec.schedule,
                spec.sign,
                resolver,
            )
        for start in range(0, len(prompts), batch_prompts):
            chunk = [
                (str(i), prompt)
                for i, prompt in enumerate(
                    prompts[start : start + batch_prompts], start
                )
                if (condition, str(i)) not in done
            ]
            if not chunk:
                continue
            params_type = SamplingParams
            if params_type is None:
                params_type = importlib.import_module("vllm").SamplingParams
            outputs = llm.generate(
                [prompt for _, prompt in chunk],
                params_type(max_tokens=1, temperature=0.0, logprobs=20),
            )
            records = []
            for (item_id, _), output in zip(chunk, outputs):
                values: dict[str, float] = {}
                for token_id, value in output.outputs[0].logprobs[0].items():
                    letter = tokenizer.decode(int(token_id)).strip()
                    if len(letter) == 1 and "A" <= letter <= "J":
                        values[letter] = float(value.logprob)
                prediction = (
                    max(values, key=lambda letter: values[letter]) if values else None
                )
                records.append(
                    {
                        "condition": condition,
                        "id": item_id,
                        "pred_letter": prediction,
                        "logprobs": values,
                    }
                )
            append_jsonl(path, records)
    finally:
        if detacher is not None:
            detacher()


def _paths(work: Path, name: str) -> Path:
    return work / "checkpoints" / name


def _letter_accuracy(
    records: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> float:
    pairs = [
        (
            parse_answer_letter(str(row.get("text", ""))),
            records[int(row["id"])]["answer_letter"],
        )
        for row in rows
    ]
    return sum(pred == truth for pred, truth in pairs) / max(1, len(pairs))


def analyze(
    cfg: dict[str, Any],
    work: Path = ROOT,
    limit: int | None = None,
    stage: str = "both",
) -> None:
    conditions = validation_conditions(cfg, limit)
    val_scores: dict[str, Any] = {}
    val_judge = read_jsonl(_paths(work, "exp3_val_judge.jsonl"))
    val_truth = load_mmlu_pro(str(cfg["validation"]["mmlu_source"]))
    base_rows = read_jsonl(_paths(work, "exp3_val_baseline_mmlu.jsonl"))
    a0 = _letter_accuracy(val_truth, base_rows)
    selection: dict[str, Any] = {}
    for schedule in cfg["schedules"]:
        group = [c for c in conditions if c.schedule == schedule]
        points = []
        for condition in group:
            judged = [row for row in val_judge if row["condition"] == condition.name]
            clean_judged = [row for row in judged if row["safe"] is not None]
            answers_rows = read_jsonl(
                _paths(work, f"exp3_val_{condition.name}_mmlu.jsonl")
            )
            score = (
                sum(row["safe"] for row in clean_judged) / max(1, len(clean_judged)),
                _letter_accuracy(val_truth, answers_rows),
            )
            blocked = sum(bool(row.get("blocked", False)) for row in judged)
            val_scores[condition.name] = {
                "p_safe": score[0],
                "a_mmlu": score[1],
                "blocked": blocked,
            }
            if blocked:
                print(
                    f"warning: {condition.name} has {blocked} blocked judge items",
                    flush=True,
                )
            points.append(score)
        index, fallback = choose_selection(
            points, a0, cfg["selection"]["capability_cap"]
        )
        selected = group[index]
        selection[schedule] = {
            "layer": selected.layer,
            "alpha": selected.alpha,
            **val_scores[selected.name],
            "fallback": fallback,
        }
    write_json_atomic(_paths(work, "exp3_selection.json"), selection)
    write_json_atomic(_paths(work, "exp3_val_scores.json"), val_scores)
    if stage == "selection":
        return
    test_judge = read_jsonl(_paths(work, "exp3_test_judge.jsonl"))
    test_truth = load_mmlu_pro("test")
    results = {
        "baseline": {
            "p_safe": mean(
                [
                    r["safe"]
                    for r in test_judge
                    if r["condition"] == "baseline" and r["safe"] is not None
                ]
            ),
            "a_mmlu": _letter_accuracy(
                test_truth, read_jsonl(_paths(work, "exp3_test_baseline_mmlu.jsonl"))
            ),
            "blocked": sum(
                bool(r.get("blocked", False))
                for r in test_judge
                if r["condition"] == "baseline"
            ),
        }
    }
    if results["baseline"]["blocked"]:
        print(
            f"warning: baseline has {results['baseline']['blocked']} blocked judge items",
            flush=True,
        )
    for schedule in cfg["schedules"]:
        name = schedule
        rows = [r for r in test_judge if r["condition"] == name]
        results[name] = {
            "p_safe": mean([r["safe"] for r in rows if r["safe"] is not None]),
            "a_mmlu": _letter_accuracy(
                test_truth,
                read_jsonl(_paths(work, f"exp3_test_{name}_mmlu.jsonl")),
            ),
            "blocked": sum(bool(r.get("blocked", False)) for r in rows),
        }
        if results[name]["blocked"]:
            print(
                f"warning: {name} has {results[name]['blocked']} blocked judge items",
                flush=True,
            )
    write_json_atomic(_paths(work, "exp3_results.json"), results)
    make_pdf(work / "figs" / "exp3_test_bars.pdf", results)


def answers_for(work: Path, condition: str, dataset: str) -> list[dict[str, Any]]:
    return read_jsonl(_paths(work, f"exp3_val_{condition}_{dataset}.jsonl"))


def mean(values: list[Any]) -> float:
    return float(sum(bool(value) for value in values) / max(1, len(values)))


def make_pdf(path: Path, values: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(values)
    fig, ax = plt.subplots()
    ax.bar(names, [values[name]["p_safe"] for name in names])
    fig.savefig(path, format="pdf")
    plt.close(fig)


def direction_phase(
    cfg: dict[str, Any], work: Path, batch_prompts: int, limit: int | None = None
) -> None:
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    pos = [
        chat_prompt(tokenizer, item, False)
        for item in load_llm_lat(
            cfg["direction"]["pos_dataset"], cfg["direction"]["n_per_class"]
        )
    ]
    neg = [
        chat_prompt(tokenizer, item, False)
        for item in load_llm_lat(
            cfg["direction"]["neg_dataset"], cfg["direction"]["n_per_class"]
        )
    ]
    save_directions(
        _paths(work, "exp3_directions.json"),
        build_dim_directions(llm, pos, neg, _smoke_layers(cfg, limit), batch_prompts),
    )


def _run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp3.yaml"))
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["direction", "generate", "judge", "analyze", "all"],
        default=["all"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-prompts", type=int, default=256)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    work = ROOT
    verify_manifest(_paths(work, "exp3_manifest.json"), cfg)
    write_manifest(_paths(work, "exp3_manifest.json"), cfg)
    with tee_stdout(work / "logs" / "exp3.log"):
        if "all" in args.phase:
            direction_phase(cfg, work, args.batch_prompts, args.limit)
            generate_validation_phase(cfg, work, args.limit, args.batch_prompts)
            judge_validation_phase(cfg, work, args.limit)
            analyze_selection(cfg, work, args.limit)
            generate_test_phase(cfg, work, args.limit, args.batch_prompts)
            judge_test_phase(cfg, work, args.limit)
            analyze_results(cfg, work, args.limit)
            return
        if "direction" in args.phase:
            direction_phase(cfg, work, args.batch_prompts, args.limit)
        if "generate" in args.phase:
            generate_phase(cfg, work, args.limit, args.batch_prompts)
        if "judge" in args.phase:
            judge_phase(cfg, work, args.limit)
        if "analyze" in args.phase:
            analyze(cfg, work, args.limit)


def generate_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    generate_validation_phase(cfg, work, limit, batch_prompts)
    if read_json(_paths(work, "exp3_selection.json")) is None:
        print("generate: selection is missing; skipping test generation", flush=True)
        return
    generate_test_phase(cfg, work, limit, batch_prompts)


def generate_validation_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    directions = load_directions(_paths(work, "exp3_directions.json"))
    conditions = validation_conditions(cfg, limit)
    val_hb, _ = harmbench_split(int(cfg["validation"]["harmbench_n"]), int(cfg["seed"]))
    hb_records = load_harmbench()
    hb_prompts = [
        chat_prompt(tokenizer, str(hb_records[i]["behavior"]), False) for i in val_hb
    ]
    mmlu_records = load_mmlu_pro(str(cfg["validation"]["mmlu_source"]))
    if limit is not None:
        mmlu_records = mmlu_records[:limit]
    mmlu_prompts = [
        chat_prompt(
            tokenizer,
            mmlu_prompt(str(row["question"]), cast(list[str], row["options"])),
            True,
        )
        for row in mmlu_records
    ]
    for name in ["baseline"] + [condition.name for condition in conditions]:
        spec = None
        if name != "baseline":
            condition = next(item for item in conditions if item.name == name)
            record = directions[condition.layer]
            spec = SteeringSpec(
                condition.layer,
                record.direction,
                condition.alpha,
                record.mean_norm,
                schedule_for(condition.schedule),
            )
        generate_jsonl(
            llm,
            hb_prompts,
            _paths(work, f"exp3_val_{name}_hb.jsonl"),
            spec,
            int(cfg["generation"]["harmbench"]["max_new_tokens"]),
            batch_prompts,
            [str(i) for i in val_hb],
        )
        generate_jsonl(
            llm,
            mmlu_prompts,
            _paths(work, f"exp3_val_{name}_mmlu.jsonl"),
            spec,
            int(cfg["validation"]["max_new_tokens"]),
            batch_prompts,
            [str(i) for i in range(len(mmlu_records))],
        )


def generate_test_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    selection = require_selection(_paths(work, "exp3_selection.json"))
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    directions = load_directions(_paths(work, "exp3_directions.json"))
    _, test_hb = harmbench_split(
        int(cfg["validation"]["harmbench_n"]), int(cfg["seed"])
    )
    hb_records = load_harmbench()
    hb_prompts = [
        chat_prompt(tokenizer, str(hb_records[i]["behavior"]), False) for i in test_hb
    ]
    mmlu_records = load_mmlu_pro("test")
    if limit is not None:
        mmlu_records = mmlu_records[:limit]
    mmlu_prompts = [
        chat_prompt(
            tokenizer,
            mmlu_prompt(str(row["question"]), cast(list[str], row["options"])),
            True,
        )
        for row in mmlu_records
    ]
    for name in ["baseline"] + list(selection):
        spec = None
        if name != "baseline":
            selected = selection[name]
            record = directions[int(selected["layer"])]
            spec = SteeringSpec(
                int(selected["layer"]),
                record.direction,
                float(selected["alpha"]),
                record.mean_norm,
                schedule_for(name),
            )
        generate_jsonl(
            llm,
            hb_prompts,
            _paths(work, f"exp3_test_{name}_hb.jsonl"),
            spec,
            int(cfg["generation"]["harmbench"]["max_new_tokens"]),
            batch_prompts,
            [str(i) for i in test_hb],
        )
        generate_jsonl(
            llm,
            mmlu_prompts,
            _paths(work, f"exp3_test_{name}_mmlu.jsonl"),
            spec,
            int(cfg["generation"]["mmlu"]["max_new_tokens"]),
            batch_prompts,
            [str(i) for i in range(len(mmlu_records))],
        )


def judge_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    judge = GeminiJudge(model=cfg["judge"]["model"], region=cfg["judge"]["region"])
    conditions = validation_conditions(cfg, limit)
    for condition in ["baseline"] + [item.name for item in conditions]:
        path = _paths(work, f"exp3_val_{condition}_hb.jsonl")
        if path.exists():
            rows = read_jsonl(path)
            judge_jsonl(
                path,
                _paths(work, "exp3_val_judge.jsonl"),
                [str(row["id"]) for row in rows],
                lambda row: {
                    "condition": condition,
                    "i": int(row["id"]),
                    "safe": judge.judge_safety(str(row.get("prompt", "")), row["text"]),
                },
                output_id=lambda row, name=condition: f"{name}/{row['id']}",
                blocked_result=lambda row: {
                    "condition": condition,
                    "i": int(row["id"]),
                    "safe": None,
                },
            )


def judge_validation_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    judge_phase(cfg, work, limit)


def judge_test_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    judge = GeminiJudge(model=cfg["judge"]["model"], region=cfg["judge"]["region"])
    selection = read_json(_paths(work, "exp3_selection.json"), {})
    conditions = (
        ["baseline"] + list(selection) if isinstance(selection, dict) else ["baseline"]
    )
    for condition in conditions:
        path = _paths(work, f"exp3_test_{condition}_hb.jsonl")
        if not path.exists():
            print(f"judge: skipping missing {path}", flush=True)
            continue
        rows = read_jsonl(path)
        judge_jsonl(
            path,
            _paths(work, "exp3_test_judge.jsonl"),
            [str(row["id"]) for row in rows],
            lambda row, name=condition: {
                "condition": name,
                "safe": judge.judge_safety("", row["text"]),
            },
            output_id=lambda row, name=condition: f"{name}/{row['id']}",
            blocked_result=lambda row, name=condition: {
                "condition": name,
                "safe": None,
            },
        )


def analyze_selection(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    analyze(cfg, work, limit, stage="selection")


def analyze_results(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    analyze(cfg, work, limit, stage="results")


def main(argv: list[str] | None = None) -> None:
    with notify_on_exit("exp3", log_file=ROOT / "logs" / "exp3.log"):
        _run(argv)


if __name__ == "__main__":
    main()
