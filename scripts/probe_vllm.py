# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

import traceback
from typing import Any

import torch

from prefix import vllm_steering as steering


PROMPTS = ["The capital of France is", "One plus one equals", "A rainbow has"]
llm: Any = None


def describe(value: Any, name: str) -> None:
    print(name, "type=", type(value), "attrs=", dir(value))


def scheduler_items(scheduler: Any) -> list[Any]:
    if isinstance(scheduler, (list, tuple)):
        return list(scheduler)
    for field in ("running", "running_requests", "running_reqs", "requests"):
        value = getattr(scheduler, field, None)
        if isinstance(value, (list, tuple)):
            return list(value)
        if value is not None and hasattr(value, "values"):
            return list(value.values())
    return []


def request_id(request: Any) -> Any:
    for field in ("request_id", "req_id", "id"):
        value = getattr(request, field, None)
        if value is not None:
            return value
    return None


def output_ids(request: Any) -> list[Any]:
    for field in ("output_token_ids", "generated_token_ids", "output_ids"):
        value = getattr(request, field, None)
        if value is not None:
            return list(value)
    return []


def main() -> None:
    global llm
    llm = steering.make_llm(
        "Qwen/Qwen3-4B", gpu_memory_utilization=0.75, max_model_len=8192
    )
    core = llm.llm_engine.engine_core.engine_core
    describe(core, "engine_core")
    scheduler = getattr(core, "scheduler", None)
    describe(scheduler, "scheduler")
    schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
    for index, item in enumerate(schedulers):
        if item is not None:
            describe(item, f"scheduler[{index}]")

    sink = steering.CaptureSink()
    original_metadata = steering._current_attn_metadata

    def observed_metadata() -> Any:
        metadata = original_metadata()
        if metadata is not None:
            qsl = getattr(metadata, "query_start_loc", None)
            computed = getattr(metadata, "num_computed_tokens", None)
            print(
                "attn_metadata",
                "query_start_loc_shape=",
                tuple(qsl.shape) if torch.is_tensor(qsl) else None,
                "num_computed_tokens=",
                computed,
            )
        return metadata

    steering._current_attn_metadata = observed_metadata

    def resolver(metadata: Any) -> list[int] | None:
        rows = steering.decode_rows(metadata) or []
        running = scheduler_items(scheduler)
        for row, request in zip(rows, running):
            ids = output_ids(request)
            print(
                "mapping",
                "row=",
                row,
                "request_id=",
                request_id(request),
                "len(output_token_ids)=",
                len(ids),
            )
        return [len(output_ids(request)) for request in running[: len(rows)]]

    steering.attach_capture(llm, 0, sink, decode_index_resolver=resolver)
    outputs = steering.generate_greedy(llm, PROMPTS, max_tokens=8)
    print("outputs=", outputs)
    print("captured=", len(sink.rows))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("probe exception:", repr(exc))
        traceback.print_exc()
        try:
            print("llm attrs=", dir(llm))
            core = llm.llm_engine.engine_core.engine_core
            print("engine_core attrs=", dir(core))
            print("scheduler attrs=", dir(getattr(core, "scheduler", None)))
        except Exception as diagnostic_error:
            print("attribute tree failure:", repr(diagnostic_error))
