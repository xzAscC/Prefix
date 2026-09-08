# pyright: reportMissingImports=false
"""Lazy vLLM steering and hidden-state capture.

LLM construction requires ``enforce_eager=True``, ``enable_chunked_prefill=False``,
and ``enable_prefix_caching=False``.  The module sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0``
and ``VLLM_USE_FLASHINFER_SAMPLER=0`` before any lazy vLLM import.  Prefix decode
interventions need a caller-injected decode-index resolver; ``scripts/probe_vllm.py``
provides diagnostics for determining that mapping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from .steering import SteeringSchedule

Detacher = Callable[[], None]
DecodeIndexResolver = Callable[[Any], list[int] | None]


@dataclass
class CaptureSink:
    rows: list[dict[str, Any]] = field(default_factory=list)


def make_llm(model_id: str, **kwargs: Any) -> Any:
    from vllm import LLM  # type: ignore[import-not-found]

    options = {
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        **kwargs,
    }
    return LLM(model=model_id, **options)


def prefill_last_rows(metadata: Any) -> list[int] | None:
    qsl = getattr(metadata, "query_start_loc", None)
    if not (qsl is not None and torch.is_tensor(qsl) and len(qsl) > 1):
        return None
    computed = getattr(metadata, "num_computed_tokens", None)
    rows: list[int] = []
    for i, length in enumerate(torch.diff(qsl)):
        if int(length) > 1 and (computed is None or int(computed[i]) == 0):
            rows.append(int(qsl[i + 1]) - 1)
    return rows or None


def decode_rows(metadata: Any) -> list[int] | None:
    qsl = getattr(metadata, "query_start_loc", None)
    computed = getattr(metadata, "num_computed_tokens", None)
    if not (
        qsl is not None
        and torch.is_tensor(qsl)
        and len(qsl) > 1
        and computed is not None
    ):
        return None
    rows: list[int] = []
    for i, length in enumerate(torch.diff(qsl)):
        if int(length) == 1 and int(computed[i]) > 0:
            rows.append(int(qsl[i]))
    return rows or None


def _engine_model(llm: Any) -> Any:
    executor = llm.llm_engine.engine_core.engine_core.model_executor
    return executor.driver_worker.worker.model_runner.model


def _current_attn_metadata() -> Any:
    from vllm.forward_context import get_forward_context  # type: ignore[import-not-found]

    metadata = getattr(get_forward_context(), "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values())) if metadata else None
    return metadata


def _block(target: Any, layer: int) -> Any:
    if hasattr(target, "forward"):
        return target
    return _engine_model(target).model.layers[layer]


def attach_steering(
    llm: Any,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    mean_norm: float,
    schedule: SteeringSchedule,
    sign: float = 1.0,
    decode_index_resolver: DecodeIndexResolver | None = None,
) -> Detacher:
    if schedule.kind == "prefix" and decode_index_resolver is None:
        raise RuntimeError("prefix steering requires a decode-index resolver")
    resolver = decode_index_resolver
    block = _block(llm, layer)
    original = block.forward
    beta = sign * alpha * mean_norm

    def steered_forward(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        metadata = _current_attn_metadata()
        prefill = prefill_last_rows(metadata)
        decode = decode_rows(metadata)
        if prefill or decode:
            out = out.clone()
            edit = direction.to(device=out.device, dtype=out.dtype) * beta
            for row in prefill or []:
                if schedule.intervene_on_prefill():
                    out[row] = out[row] + edit
            if decode and schedule.kind == "full":
                for row in decode:
                    out[row] = out[row] + edit
            elif decode and schedule.kind == "prefix":
                indices = resolver(metadata) if resolver is not None else None
                if indices is not None:
                    for row, k in zip(decode, indices):
                        if schedule.intervene_on_decode(k):
                            out[row] = out[row] + edit
        return out

    block.forward = steered_forward

    def detach() -> None:
        block.forward = original

    return detach


def attach_capture(
    llm: Any,
    layer: int,
    sink: CaptureSink,
    decode_index_resolver: DecodeIndexResolver | None = None,
) -> Detacher:
    block = _block(llm, layer)
    original = block.forward

    def capturing_forward(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        metadata = _current_attn_metadata()
        prefill = prefill_last_rows(metadata) or []
        decode = decode_rows(metadata) or []
        indices = decode_index_resolver(metadata) if decode_index_resolver else None
        for row in prefill:
            slot = _row_slot(metadata, row)
            sink.rows.append(
                {
                    "phase": "prefill",
                    "row": row,
                    "k": None,
                    "slot": slot,
                    "hidden": _copy_hidden(out[row]),
                }
            )
        for position, row in enumerate(decode):
            slot = _row_slot(metadata, row)
            k = (
                indices[position]
                if indices is not None and position < len(indices)
                else None
            )
            sink.rows.append(
                {
                    "phase": "decode",
                    "row": row,
                    "k": k,
                    "slot": slot,
                    "hidden": _copy_hidden(out[row]),
                }
            )
        return out

    block.forward = capturing_forward

    def detach() -> None:
        block.forward = original

    return detach


def _copy_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.detach().to(device="cpu", dtype=torch.float32).squeeze()


def _row_slot(metadata: Any, row: int) -> int | None:
    qsl = getattr(metadata, "query_start_loc", None)
    if qsl is None or not torch.is_tensor(qsl):
        return None
    for slot in range(len(qsl) - 1):
        if int(qsl[slot]) <= row < int(qsl[slot + 1]):
            return slot
    return None


def generate_greedy(
    llm: Any, prompts: list[str], max_tokens: int, batch_prompts: int = 256
) -> list[str]:
    from vllm import SamplingParams  # type: ignore[import-not-found]

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    texts: list[str] = []
    for start in range(0, len(prompts), batch_prompts):
        outputs = llm.generate(prompts[start : start + batch_prompts], params)
        texts.extend(output.outputs[0].text for output in outputs)
    return texts


def generate_greedy_with_ids(
    llm: Any, prompts: list[str], max_tokens: int, batch_prompts: int = 256
) -> list[tuple[str, list[int], list[int]]]:
    from vllm import SamplingParams  # type: ignore[import-not-found]

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    triples: list[tuple[str, list[int], list[int]]] = []
    for start in range(0, len(prompts), batch_prompts):
        outputs = llm.generate(prompts[start : start + batch_prompts], params)
        for output in outputs:
            triples.append(
                (
                    output.outputs[0].text,
                    [int(token) for token in output.prompt_token_ids],
                    [int(token) for token in output.outputs[0].token_ids],
                )
            )
    return triples


__all__ = [
    "CaptureSink",
    "SteeringSchedule",
    "attach_capture",
    "attach_steering",
    "decode_rows",
    "generate_greedy",
    "generate_greedy_with_ids",
    "make_llm",
    "prefill_last_rows",
]
