from __future__ import annotations

import importlib
from math import exp
from types import ModuleType

import pytest


def pipeline() -> ModuleType:
    try:
        return importlib.import_module("prefix.no_steering")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing standalone no-steering production module: {error}")


def test_conditional_ppl_uses_only_generated_selected_logprobs() -> None:
    module = pipeline()
    result = module.conditional_ppl(
        [
            {
                "prompt_logprobs": [-0.01, -0.02],
                "generated_token_logprobs": [-0.5, -1.0],
                "selected_logprobs": [-0.5, -1.0],
                "unselected_logprobs": [-100.0],
            },
            {
                "prompt_logprobs": [-100.0],
                "generated_token_logprobs": [-0.25],
                "selected_logprobs": [-0.25],
                "unselected_logprobs": [-100.0],
            },
        ]
    )
    assert result == pytest.approx(exp((0.5 + 1.0 + 0.25) / 3))


def test_conditional_ppl_aggregates_total_nll_over_total_response_tokens() -> None:
    module = pipeline()
    result = module.conditional_ppl(
        [
            {"selected_logprobs": [-1.0]},
            {"selected_logprobs": [-2.0, -3.0]},
        ]
    )
    assert result == pytest.approx(exp(6.0 / 3.0))
