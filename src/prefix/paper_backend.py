"""Sequential Transformers backend for paper operators and adaptive policies.

DAS probes receive copies of the inherited KV state. Only the final chosen
intervention commits cache updates; speculative forwards never advance it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from .paper_operators import CoastOperator, das_coefficient, strength_at


@dataclass
class Intervention:
    layer: int
    direction: torch.Tensor
    operator: str = 'additive'
    policy: str = 'full'
    coefficient: float = 1.
    length: int = 5
    tau: float = 128.
    target_cosine: float = 0.
    coast: CoastOperator | None = None
    probe_weight: torch.Tensor | None = None
    probe_bias: float = 0.
    das_top_p: float = .9
    das_maximum: float = 2.
    act_amplitude: float = 12.
    act_bias: float = 0.


class PaperBackend:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.layers = model.model.layers
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, model_id: str, revision: str, device: str = 'cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        dtype = torch.float32 if device == 'cpu' else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                    torch_dtype=dtype).to(device)
        return cls(model, tokenizer)

    def encode(self, messages: list[dict[str, str]], *, thinking: bool) -> torch.Tensor:
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=thinking, return_tensors='pt')
        return tokens.to(self.device)

    @torch.inference_mode()
    def capture(self, messages: list[dict[str, str]], layers: list[int], *, thinking: bool):
        result = {}
        handles = []
        for index in layers:
            def save(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                result[index] = hidden[0, -1].float().cpu()
            handles.append(self.layers[index].register_forward_hook(save))
        try:
            self.model(self.encode(messages, thinking=thinking), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return result

    @torch.inference_mode()
    def generate_tokens(self, input_ids: torch.Tensor, max_tokens: int,
                        spec: Intervention | None = None, *, eos_ids: list[int] | None = None):
        if max_tokens < 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('one nonempty input and positive max_tokens are required')
        if spec is not None:
            if not 0 <= spec.layer < len(self.layers):
                raise ValueError('intervention layer is out of range')
            if spec.operator not in {'additive', 'coast'}:
                raise ValueError('unknown intervention operator')
            if spec.operator == 'coast' and (spec.coast is None or spec.policy not in {'full', 'prefix'}):
                raise ValueError('COAST requires its fitted operator and a full/prefix schedule')
            if spec.policy == 'act' and spec.probe_weight is None:
                raise ValueError('ACT requires a trained classifier')
        if eos_ids is None:
            eos_ids = self.model.generation_config.eos_token_id
            eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else eos_ids)
        cache = None
        tokens, coefficients = [], []
        step_input = input_ids.to(self.device)
        for position in range(max_tokens):
            chosen = 0.

            def forward(past, override=None):
                nonlocal chosen
                handle = None
                if spec is not None:
                    def edit(module, args, output):
                        nonlocal chosen
                        hidden = output[0] if isinstance(output, tuple) else output
                        current = hidden[0, -1]
                        probability = None
                        if spec.policy == 'act':
                            probability = float(torch.sigmoid(current.float() @ spec.probe_weight.to(current.device).float() + spec.probe_bias))
                        chosen = override if override is not None else strength_at(
                            spec.policy, position, spec.coefficient, length=spec.length,
                            tau=spec.tau, concept_probability=probability,
                            act_amplitude=spec.act_amplitude, act_bias=spec.act_bias)
                        updated = hidden.clone()
                        if spec.operator == 'coast':
                            if spec.policy == 'full' or position < spec.length:
                                updated[0, -1] = spec.coast(current, spec.target_cosine)
                        else:
                            updated[0, -1] = current + chosen * spec.direction.to(current)
                        return (updated, *output[1:]) if isinstance(output, tuple) else updated
                    handle = self.layers[spec.layer].register_forward_hook(edit)
                try:
                    return self.model(step_input, past_key_values=past, use_cache=True)
                finally:
                    if handle is not None:
                        handle.remove()

            coefficient = None
            if spec is not None and spec.policy == 'das':
                base = forward(copy.deepcopy(cache), override=0.).logits[0, -1].float().clone()
                probe = forward(copy.deepcopy(cache), override=spec.das_maximum).logits[0, -1].float().clone()
                coefficient = das_coefficient(base, probe, top_p=spec.das_top_p, maximum=spec.das_maximum)
            output = forward(cache, override=coefficient)
            cache = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            tokens.append(token)
            coefficients.append(float(chosen))
            if token in eos_ids:
                break
            step_input = torch.tensor([[token]], device=self.device)
        return {'token_ids': tokens, 'coefficients': coefficients,
                'finish_reason': 'eos' if tokens[-1] in eos_ids else 'length'}

    def generate(self, messages: list[dict[str, str]], max_tokens: int,
                 spec: Intervention | None = None, *, thinking: bool = False):
        output = self.generate_tokens(self.encode(messages, thinking=thinking), max_tokens, spec)
        output['response'] = self.tokenizer.decode(output['token_ids'], skip_special_tokens=True)
        return output
