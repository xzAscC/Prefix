# pyright: reportMissingImports=false
"""Lazy vLLM Qwen3 residual steering and capture.

The metadata helpers classify rows from ``query_start_loc`` and ``seq_lens``;
they do not depend on ``num_computed_tokens``.  ``make_decode_index_resolver``
returns a lazy production resolver that reads ``model_runner.input_batch`` and
returns one-based decode indices in decode-row order.  Capture supports either
full residual vectors or GPU-computed scalar projections.
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
DecodeIndexResolver = Callable[[Any], list[int]]


@dataclass
class CaptureSink:
    rows: list[dict[str, Any]] = field(default_factory=list)


def make_llm(model_id: str, **kwargs: Any) -> Any:
    from vllm import LLM  # type: ignore[import-not-found]

    options: dict[str, Any] = {
        **kwargs,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    return LLM(model=model_id, **options)


def _row_info(metadata: Any) -> tuple[list[int], list[int], list[int], list[int]]:
    qsl = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if not (torch.is_tensor(qsl) and torch.is_tensor(seq_lens) and qsl.numel() > 1):
        return [], [], [], []
    values = torch.stack((qsl[:-1], qsl[1:], seq_lens), dim=1).to("cpu").tolist()
    prefill: list[int] = []
    decode: list[int] = []
    slots: list[int] = []
    decode_slots: list[int] = []
    for slot, (start, end, seq_len) in enumerate(values):
        query_len = end - start
        if query_len > 1:
            prefill.append(end - 1)
            slots.append(slot)
        elif query_len == 1 and seq_len == 1:
            prefill.append(start)
            slots.append(slot)
        elif query_len == 1 and seq_len > 1:
            decode.append(start)
            decode_slots.append(slot)
    return prefill, decode, slots, decode_slots


def prefill_last_rows(metadata: Any) -> list[int] | None:
    rows, _, _, _ = _row_info(metadata)
    return rows or None


def decode_rows(metadata: Any) -> list[int] | None:
    _, rows, _, _ = _row_info(metadata)
    return rows or None


def _model_runner(llm: Any) -> Any:
    return llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner


def _engine_model(llm: Any) -> Any:
    return _model_runner(llm).model


_LATEST_INPUT_BATCH: dict[int, Any] = {}
_PREPARE_HOOK_KEYS: set[int] = set()


def _ensure_prepare_inputs_hook(llm: Any) -> None:
    key = id(llm)
    if key in _PREPARE_HOOK_KEYS:
        return
    try:
        runner = _model_runner(llm)
    except AttributeError:
        return
    original = getattr(runner, "prepare_inputs", None)
    if original is None:
        raise RuntimeError("model runner has no prepare_inputs to wrap")

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        _LATEST_INPUT_BATCH[key] = result
        return result

    runner.prepare_inputs = wrapper
    _PREPARE_HOOK_KEYS.add(key)


def _normalize_req_id(req_id: str) -> str:
    head = req_id.rpartition("-")[0]
    return head or req_id


def _input_batch(llm: Any) -> Any:
    batch = _LATEST_INPUT_BATCH.get(id(llm))
    if batch is None:
        raise RuntimeError("no live input batch for this engine")
    return batch


def make_decode_index_resolver(llm: Any) -> DecodeIndexResolver:
    _ensure_prepare_inputs_hook(llm)

    def resolve(metadata: Any) -> list[int]:
        qsl = getattr(metadata, "query_start_loc", None)
        seq_lens = getattr(metadata, "seq_lens", None)
        if not (torch.is_tensor(qsl) and torch.is_tensor(seq_lens) and qsl.numel() > 1):
            return []
        input_batch = _input_batch(llm)
        request_ids = input_batch.req_ids
        expected = qsl.numel() - 1
        if len(request_ids) != expected:
            raise RuntimeError(
                f"input batch req_ids length {len(request_ids)} does not match qsl requests {expected}"
            )
        prefill_lens = input_batch.prefill_len_np
        if len(prefill_lens) != expected:
            raise RuntimeError(
                "input batch prefill_len_np length does not match qsl requests"
            )
        values = torch.stack((qsl[:-1], qsl[1:], seq_lens), dim=1).to("cpu").tolist()
        prompt_lens = [int(length) for length in prefill_lens]
        result: list[int] = []
        for slot, (start, end, seq_len) in enumerate(values):
            if end - start == 1 and seq_len > 1:
                k = seq_len - prompt_lens[slot]
                if k < 1:
                    raise RuntimeError("decode index must be at least 1")
                result.append(k)
        return result

    return resolve


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


def _hidden_output(out: Any) -> torch.Tensor:
    if isinstance(out, tuple) and len(out) == 2:
        return out[0] + out[1]
    if not torch.is_tensor(out):
        raise TypeError("decoder output must be a tensor or a two-tensor tuple")
    return out


def _steered_output(out: Any, hidden: torch.Tensor) -> Any:
    if isinstance(out, tuple):
        return (hidden, *out[1:])
    return hidden


def _request_ids(llm: Any, count: int) -> list[str | None]:
    request_ids = _input_batch(llm).req_ids
    if len(request_ids) != count:
        raise RuntimeError(
            "input batch req_ids length does not match metadata requests"
        )
    return [_normalize_req_id(str(req_id)) for req_id in request_ids]


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
    block = _block(llm, layer)
    if not hasattr(llm, "forward"):
        _ensure_prepare_inputs_hook(llm)
    original = block.forward
    beta = sign * alpha * mean_norm
    edit_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def steered_forward(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        metadata = _current_attn_metadata()
        prefill, decode, _, _ = _row_info(metadata)
        if not prefill and not decode:
            return out
        rows = list(prefill) if schedule.intervene_on_prefill() else []
        if decode:
            if schedule.kind == "full":
                rows.extend(decode)
            elif schedule.kind == "prefix":
                assert decode_index_resolver is not None
                indices = decode_index_resolver(metadata)
                if len(indices) != len(decode):
                    raise RuntimeError("decode-index resolver result length mismatch")
                rows.extend(
                    row
                    for row, k in zip(decode, indices)
                    if schedule.intervene_on_decode(k)
                )
        if not rows:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        hidden = hidden.clone()
        key = (hidden.device, hidden.dtype)
        edit = edit_cache.get(key)
        if edit is None:
            edit = direction.to(device=hidden.device, dtype=hidden.dtype) * beta
            edit_cache[key] = edit
        row_index = torch.tensor(rows, device=hidden.device, dtype=torch.long)
        hidden.index_add_(0, row_index, edit.expand(len(rows), -1))
        return _steered_output(out, hidden)

    block.forward = steered_forward

    def detach() -> None:
        block.forward = original

    return detach


def attach_capture(
    llm: Any,
    layer: int,
    sink: CaptureSink,
    decode_index_resolver: DecodeIndexResolver | None = None,
    scalar_directions: list[torch.Tensor] | None = None,
) -> Detacher:
    block = _block(llm, layer)
    if not hasattr(llm, "forward"):
        _ensure_prepare_inputs_hook(llm)
    original = block.forward

    def capturing_forward(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        hidden = _hidden_output(out)
        metadata = _current_attn_metadata()
        prefill, decode, prefill_slots, decode_slots = _row_info(metadata)
        indices = decode_index_resolver(metadata) if decode_index_resolver else None
        if indices is not None and len(indices) != len(decode):
            raise RuntimeError("decode-index resolver result length mismatch")
        request_ids = _request_ids(llm, len(prefill_slots) + len(decode_slots))
        rows = prefill + decode
        slots = prefill_slots + decode_slots
        if not rows:
            return out
        row_index = torch.tensor(rows, device=hidden.device, dtype=torch.long)
        selected = hidden.index_select(0, row_index)
        if scalar_directions is None:
            values = selected.detach().to(device="cpu", dtype=torch.float32)
            for position, (row, slot) in enumerate(zip(rows, slots)):
                sink.rows.append(
                    {
                        "phase": "prefill" if position < len(prefill) else "decode",
                        "row": row,
                        "k": None
                        if position < len(prefill)
                        else indices[position - len(prefill)]
                        if indices
                        else None,
                        "slot": slot,
                        "request_id": request_ids[slot],
                        "hidden": values[position].squeeze(),
                    }
                )
        else:
            directions = torch.stack(
                [
                    direction.to(device=hidden.device, dtype=hidden.dtype)
                    for direction in scalar_directions
                ]
            )
            dots = selected @ directions.T
            norms = torch.linalg.vector_norm(selected, dim=1, keepdim=True)
            values = (
                torch.cat((dots, norms), dim=1)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .tolist()
            )
            for position, (row, slot, value) in enumerate(zip(rows, slots, values)):
                sink.rows.append(
                    {
                        "phase": "prefill" if position < len(prefill) else "decode",
                        "slot": slot,
                        "request_id": request_ids[slot],
                        "k": None
                        if position < len(prefill)
                        else indices[position - len(prefill)]
                        if indices
                        else None,
                        "dots": value[:-1],
                        "norm": value[-1],
                    }
                )
        return out

    block.forward = capturing_forward

    def detach() -> None:
        block.forward = original

    return detach


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
    "make_decode_index_resolver",
    "make_llm",
    "prefill_last_rows",
]
