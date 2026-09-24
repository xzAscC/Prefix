"""Fixed-context comparisons using the real Transformers attention module."""
from __future__ import annotations

import copy

import torch


class FixedNativeAttention:
    """Replay a pretrained attention module on measured, frozen representations.

    The readout is after all selected positions, including the 128-position arm.
    Thus its Q projection, QK normalization, and RoPE are identical in every arm.
    FP32 arithmetic preserves the small shifts in the 0.001-strength condition;
    pretrained weights are copied unchanged from the BF16 model and upcast.
    """
    def __init__(self, attention, hidden, position_embeddings, prompt_length, head):
        self.attn = copy.deepcopy(attention).float().eval()
        self.attn.config = copy.copy(self.attn.config)
        self.attn.config._attn_implementation = 'eager'
        self.hidden = hidden.detach().clone().float()
        self.embeddings = tuple(t.detach().float() for t in position_embeddings)
        self.prompt_length, self.head = prompt_length, head
        if not 1 <= prompt_length <= len(hidden):
            raise ValueError('invalid prompt length')
        self.mask = torch.full((1, 1, len(hidden), len(hidden)), -torch.inf,
                               dtype=torch.float32, device=hidden.device).triu(1)
        window = getattr(self.attn, 'sliding_window', None)
        if window is not None:
            positions = torch.arange(len(hidden), device=hidden.device)
            self.mask.masked_fill_(positions[None] <= positions[:, None] - window, -torch.inf)

    @torch.inference_mode()
    def output(self, direction, length, strength):
        start = self.prompt_length - 1
        if length < 0 or start + length > len(self.hidden) - 1:
            raise ValueError('intervention must exclude the fixed readout query')
        changed = self.hidden.clone()
        if length:
            changed[start:start+length] += strength * direction.to(changed)
        captured = []
        head_dim = self.attn.head_dim
        def capture(_module, args):
            captured.append(args[0][0,-1,self.head*head_dim:(self.head+1)*head_dim].detach().clone())
        handle = self.attn.o_proj.register_forward_pre_hook(capture)
        try:
            _, weights = self.attn(changed[None], self.embeddings, self.mask)
        finally:
            handle.remove()
        if not torch.isfinite(captured[0]).all():
            raise ValueError('nonfinite native output')
        return captured[0], weights[0,self.head,-1].detach()
