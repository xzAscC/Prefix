from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from prefix.data import (
    direction_prompt,
    load_math500,
    math500_partition,
    neutral_math_prompt,
)
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

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    sign: str
    schedule: str
    layer: int
    alpha: float

    @property
    def name(self) -> str:
        return f"{self.sign}_{self.schedule}_l{self.layer}_a{repr(self.alpha)}"


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
        Condition(sign, schedule, int(layer), float(alpha))
        for sign in ("pos", "neg")
        for schedule in cfg["schedules"]
        for layer in layers
        for alpha in grids[schedule]
    ]


def choose_selection(
    points: list[tuple[float, float]], baseline: float, cap: float
) -> tuple[int, bool]:
    frontier = pareto_frontier(points)
    try:
        index = select_operating_point([points[i] for i in frontier], baseline, cap)
        return frontier[index], False
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
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    output_id: Callable[[dict[str, Any]], str] | None = None,
    blocked_result: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    require_complete(generation, expected, label=str(generation))
    done = completed_ids(output)
    rows = [row for row in read_jsonl(generation) if str(row["id"]) not in done]

    def judge_item(row: dict[str, Any]) -> dict[str, Any]:
        try:
            result = judge(row)
        except JudgeBlocked:
            result: dict[str, Any] = (
                blocked_result(row)
                if blocked_result
                else {
                    "format_boxed": None,
                    "format_answer_is": None,
                    "answer_correct": None,
                }
            )
            result["blocked"] = True
            return result
        except RuntimeError:
            result = {
                "format_boxed": None,
                "format_answer_is": None,
                "answer_correct": None,
                "blocked": False,
                "unparseable": True,
            }
            return result
        result["blocked"] = False
        return result

    results = judge_batch(judge_item, rows, max_workers=8)
    append_jsonl(
        output,
        [
            {"id": output_id(row) if output_id else str(row["id"]), **result}
            for row, result in zip(rows, results)
        ],
    )


def steer_success(sign: str, result: dict[str, bool]) -> bool:
    return bool(result["format_boxed"] if sign == "pos" else result["format_answer_is"])


def unblocked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("answer_correct") is not None]


def blocked_count(rows: list[dict[str, Any]]) -> int:
    return sum(bool(row.get("blocked", False)) for row in rows)


def _path(work: Path, name: str) -> Path:
    return work / "checkpoints" / name


def _partition(cfg: dict[str, Any]) -> dict[str, list[int]]:
    values = cfg["math500"]["partition"]
    return math500_partition(
        n_direction=int(values["n_direction"]),
        n_val=int(values["n_val"]),
        seed=int(values["seed"]),
        n_total=int(values.get("n_total", 500)),
    )


def _legacy_analyze(
    cfg: dict[str, Any], work: Path = ROOT, limit: int | None = None
) -> None:
    conditions = validation_conditions(cfg, limit)
    judges = read_jsonl(_path(work, "exp4_val_judge.jsonl"))
    baseline = [r for r in judges if r.get("condition") == "baseline"]
    clean_baseline = unblocked(baseline)
    baseline_math = sum(r["answer_correct"] for r in clean_baseline) / max(
        1, len(clean_baseline)
    )
    scores: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for sign in ("pos", "neg"):
        for schedule in cfg["schedules"]:
            group = [c for c in conditions if c.sign == sign and c.schedule == schedule]
            points = []
            for condition in group:
                rows = [r for r in judges if r.get("condition") == condition.name]
                clean_rows = unblocked(rows)
                values = (
                    sum(steer_success(sign, r) for r in clean_rows)
                    / max(1, len(clean_rows)),
                    sum(r["answer_correct"] for r in clean_rows)
                    / max(1, len(clean_rows)),
                )
                scores[condition.name] = {
                    "steer_success": values[0],
                    "a_math": values[1],
                    "blocked": blocked_count(rows),
                }
                if scores[condition.name]["blocked"]:
                    print(
                        f"warning: {condition.name} has {scores[condition.name]['blocked']} blocked judge items",
                        flush=True,
                    )
                points.append(values)
            index, fallback = choose_selection(
                points, baseline_math, cfg["selection"]["capability_cap"]
            )
            chosen = group[index]
            selection[f"{sign}/{schedule}"] = {
                "layer": chosen.layer,
                "alpha": chosen.alpha,
                **scores[chosen.name],
                "fallback": fallback,
            }
    write_json_atomic(_path(work, "exp4_selection.json"), selection)
    write_json_atomic(_path(work, "exp4_val_scores.json"), scores)
    test = read_jsonl(_path(work, "exp4_test_judge.jsonl"))
    results: dict[str, Any] = {
        "baseline": {
            "steer_success": 0.0,
            "a_math": sum(
                r["answer_correct"]
                for r in unblocked(
                    [r for r in test if r.get("condition") == "baseline"]
                )
            )
            / max(
                1,
                len(unblocked([r for r in test if r.get("condition") == "baseline"])),
            ),
            "blocked": blocked_count(
                [r for r in test if r.get("condition") == "baseline"]
            ),
        }
    }
    if results["baseline"]["blocked"]:
        print(
            f"warning: baseline has {results['baseline']['blocked']} blocked judge items",
            flush=True,
        )
    for key in selection:
        sign = key.split("/")[0]
        rows = [r for r in test if r.get("condition") == key.replace("/", "_")]
        clean_rows = unblocked(rows)
        results[key] = {
            "steer_success": sum(steer_success(sign, r) for r in clean_rows)
            / max(1, len(clean_rows)),
            "a_math": sum(r["answer_correct"] for r in clean_rows)
            / max(1, len(clean_rows)),
            "blocked": blocked_count(rows),
        }
        if results[key]["blocked"]:
            print(
                f"warning: {key} has {results[key]['blocked']} blocked judge items",
                flush=True,
            )
    write_json_atomic(_path(work, "exp4_results.json"), results)
    make_pdf(work / "figs" / "exp4_test.pdf", results)


def make_pdf(path: Path, values: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar(list(values), [value["a_math"] for value in values.values()])
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp4.yaml"))
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
    verify_manifest(_path(ROOT, "exp4_manifest.json"), cfg)
    write_manifest(_path(ROOT, "exp4_manifest.json"), cfg)
    with tee_stdout(ROOT / "logs" / "exp4.log"):
        if "all" in args.phase:
            direction_phase(cfg, ROOT, args.batch_prompts, args.limit)
            generate_validation_phase(cfg, ROOT, args.limit, args.batch_prompts)
            judge_validation_phase(cfg, ROOT, args.limit)
            analyze_selection(cfg, ROOT, args.limit)
            generate_test_phase(cfg, ROOT, args.limit, args.batch_prompts)
            judge_test_phase(cfg, ROOT, args.limit)
            analyze_results(cfg, ROOT, args.limit)
            return
        if "direction" in args.phase:
            direction_phase(cfg, ROOT, args.batch_prompts, args.limit)
        if "generate" in args.phase:
            generate_phase(cfg, ROOT, args.limit, args.batch_prompts)
        if "judge" in args.phase:
            judge_phase(cfg, ROOT, args.limit)
        if "analyze" in args.phase:
            analyze(cfg, ROOT, args.limit)


def direction_phase(
    cfg: dict[str, Any], work: Path, batch_prompts: int, limit: int | None = None
) -> None:
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    records = load_math500()
    part = _partition(cfg)
    pos = [
        chat_prompt(
            tokenizer, direction_prompt(str(records[i]["problem"]), True), False
        )
        for i in part["direction"]
    ]
    neg = [
        chat_prompt(
            tokenizer, direction_prompt(str(records[i]["problem"]), False), False
        )
        for i in part["direction"]
    ]
    save_directions(
        _path(work, "exp4_directions.json"),
        build_dim_directions(llm, pos, neg, _smoke_layers(cfg, limit), batch_prompts),
    )


def _condition_spec(condition: Condition, directions: dict[int, Any]) -> SteeringSpec:
    record = directions[condition.layer]
    return SteeringSpec(
        condition.layer,
        record.direction,
        condition.alpha,
        record.mean_norm,
        schedule_for(condition.schedule),
        1.0 if condition.sign == "pos" else -1.0,
    )


def generate_validation_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    records = load_math500()
    part = _partition(cfg)
    directions = load_directions(_path(work, "exp4_directions.json"))
    conditions = validation_conditions(cfg, limit)
    for condition_name in ["baseline"] + [item.name for item in conditions]:
        spec = None
        if condition_name != "baseline":
            item = next(c for c in conditions if c.name == condition_name)
            spec = _condition_spec(item, directions)
        prompts = [
            chat_prompt(
                tokenizer, neutral_math_prompt(str(records[i]["problem"])), True
            )
            for i in part["val"]
        ]
        generate_jsonl(
            llm,
            prompts,
            _path(work, f"exp4_val_{condition_name}.jsonl"),
            spec,
            cfg["validation"]["max_new_tokens"],
            batch_prompts,
            [str(i) for i in part["val"]],
        )


def generate_test_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    selection = require_selection(_path(work, "exp4_selection.json"))
    llm: Any = get_engine(
        cfg["model"]["id"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=float(cfg["model"].get("gpu_memory_utilization", 0.9)),
    )
    tokenizer = llm.get_tokenizer()
    records = load_math500()
    part = _partition(cfg)
    directions = load_directions(_path(work, "exp4_directions.json"))
    for condition in ["baseline"] + list(selection):
        spec = None
        if condition != "baseline":
            sign, schedule = condition.split("/")
            selected = selection[condition]
            record = directions[int(selected["layer"])]
            spec = SteeringSpec(
                int(selected["layer"]),
                record.direction,
                float(selected["alpha"]),
                record.mean_norm,
                schedule_for(schedule),
                1.0 if sign == "pos" else -1.0,
            )
        test_ids = list(part["test"])
        if limit is not None:
            test_ids = test_ids[:limit]
        prompts = [
            chat_prompt(
                tokenizer, neutral_math_prompt(str(records[i]["problem"])), True
            )
            for i in test_ids
        ]
        generate_jsonl(
            llm,
            prompts,
            _path(work, f"exp4_test_{condition.replace('/', '_')}.jsonl"),
            spec,
            cfg["generation"]["max_new_tokens"],
            batch_prompts,
            [str(i) for i in test_ids],
        )


def generate_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, batch_prompts: int
) -> None:
    generate_validation_phase(cfg, work, limit, batch_prompts)
    if read_json(_path(work, "exp4_selection.json")) is None:
        print("generate: selection is missing; skipping test generation", flush=True)
        return
    generate_test_phase(cfg, work, limit, batch_prompts)


def _judge_phase(
    cfg: dict[str, Any], work: Path, limit: int | None, phase: str, strict: bool
) -> None:
    judge = GeminiJudge(model=cfg["judge"]["model"], region=cfg["judge"]["region"])
    records = load_math500()
    if phase == "val":
        conditions = ["baseline"] + [
            item.name for item in validation_conditions(cfg, limit)
        ]
    else:
        selection = read_json(_path(work, "exp4_selection.json"))
        if selection is None:
            print("judge: selection is missing; skipping test judging", flush=True)
            return
        conditions = ["baseline"] + list(selection)
    for condition in conditions:
        path = _path(work, f"exp4_{phase}_{condition.replace('/', '_')}.jsonl")
        if not path.exists():
            if strict:
                raise RuntimeError(f"missing generation file: {path}")
            print(f"judge: skipping missing {path}", flush=True)
            continue
        rows = read_jsonl(path)
        expected = [str(row["id"]) for row in rows]
        judge_jsonl(
            path,
            _path(work, f"exp4_{phase}_judge.jsonl"),
            expected,
            lambda row, name=condition: {
                "condition": name,
                **judge.judge_math(row["text"], str(records[int(row["id"])]["answer"])),
            },
            output_id=lambda row, name=condition: f"{name}/{row['id']}",
            blocked_result=lambda row, name=condition: {
                "condition": name,
                "format_boxed": None,
                "format_answer_is": None,
                "answer_correct": None,
            },
        )


def judge_validation_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    _judge_phase(cfg, work, limit, "val", True)


def judge_test_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    _judge_phase(cfg, work, limit, "test", False)


def judge_phase(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    judge_validation_phase(cfg, work, limit)
    judge_test_phase(cfg, work, limit)


def analyze_selection(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    conditions = validation_conditions(cfg, limit)
    judges = read_jsonl(_path(work, "exp4_val_judge.jsonl"))
    baseline = [r for r in judges if r.get("condition") == "baseline"]
    clean_baseline = unblocked(baseline)
    baseline_math = sum(r["answer_correct"] for r in clean_baseline) / max(
        1, len(clean_baseline)
    )
    scores: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for sign in ("pos", "neg"):
        for schedule in cfg["schedules"]:
            group = [c for c in conditions if c.sign == sign and c.schedule == schedule]
            points = []
            for condition in group:
                rows = [r for r in judges if r.get("condition") == condition.name]
                clean_rows = unblocked(rows)
                values = (
                    sum(steer_success(sign, r) for r in clean_rows)
                    / max(1, len(clean_rows)),
                    sum(r["answer_correct"] for r in clean_rows)
                    / max(1, len(clean_rows)),
                )
                scores[condition.name] = {
                    "steer_success": values[0],
                    "a_math": values[1],
                    "blocked": blocked_count(rows),
                }
                if scores[condition.name]["blocked"]:
                    print(
                        f"warning: {condition.name} has {scores[condition.name]['blocked']} blocked judge items",
                        flush=True,
                    )
                points.append(values)
            index, fallback = choose_selection(
                points, baseline_math, cfg["selection"]["capability_cap"]
            )
            chosen = group[index]
            selection[f"{sign}/{schedule}"] = {
                "layer": chosen.layer,
                "alpha": chosen.alpha,
                **scores[chosen.name],
                "fallback": fallback,
            }
    write_json_atomic(_path(work, "exp4_selection.json"), selection)
    write_json_atomic(_path(work, "exp4_val_scores.json"), scores)


def analyze_results(cfg: dict[str, Any], work: Path, limit: int | None) -> None:
    selection = require_selection(_path(work, "exp4_selection.json"))
    test = read_jsonl(_path(work, "exp4_test_judge.jsonl"))
    results: dict[str, Any] = {
        "baseline": {
            "steer_success": 0.0,
            "a_math": sum(
                r["answer_correct"]
                for r in unblocked(
                    [r for r in test if r.get("condition") == "baseline"]
                )
            )
            / max(
                1,
                len(unblocked([r for r in test if r.get("condition") == "baseline"])),
            ),
            "blocked": blocked_count(
                [r for r in test if r.get("condition") == "baseline"]
            ),
        }
    }
    for key in selection:
        sign = key.split("/")[0]
        rows = [r for r in test if r.get("condition") == key.replace("/", "_")]
        clean_rows = unblocked(rows)
        results[key] = {
            "steer_success": sum(steer_success(sign, r) for r in clean_rows)
            / max(1, len(clean_rows)),
            "a_math": sum(r["answer_correct"] for r in clean_rows)
            / max(1, len(clean_rows)),
            "blocked": blocked_count(rows),
        }
        if results[key]["blocked"]:
            print(
                f"warning: {key} has {results[key]['blocked']} blocked judge items",
                flush=True,
            )
    write_json_atomic(_path(work, "exp4_results.json"), results)
    make_pdf(work / "figs" / "exp4_test.pdf", results)


def analyze(cfg: dict[str, Any], work: Path, limit: int | None = None) -> None:
    if read_json(_path(work, "exp4_selection.json")) is None:
        try:
            part = _partition(cfg)
            expected = [str(index) for index in part["val"]]
            conditions = ["baseline"] + [
                item.name for item in validation_conditions(cfg, limit)
            ]
            for condition in conditions:
                require_complete(
                    _path(work, f"exp4_val_{condition}.jsonl"),
                    expected,
                    label=f"validation {condition}",
                )
            analyze_selection(cfg, work, limit)
        except (FileNotFoundError, RuntimeError, KeyError) as error:
            print(f"analyze: selection unavailable: {error}", flush=True)
            return
    if read_json(_path(work, "exp4_test_judge.jsonl")) is None:
        print("analyze: test judge is missing; skipping results", flush=True)
        return
    analyze_results(cfg, work, limit)


def main(argv: list[str] | None = None) -> None:
    with notify_on_exit("exp4", log_file=ROOT / "logs" / "exp4.log"):
        _run(argv)


if __name__ == "__main__":
    main()
