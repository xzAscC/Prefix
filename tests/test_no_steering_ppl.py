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


def test_incremental_ppl_accumulation_preserves_row_order_and_token_count() -> None:
    module = pipeline()
    rows = [
        {"selected_logprobs": [-1.0e16]},
        {"selected_logprobs": [-1.0]},
        {"selected_logprobs": [-1.0]},
    ]

    total_nll = 0.0
    token_count = 0
    for row in rows:
        total_nll, token_count = module._accumulate_ppl(total_nll, token_count, row)

    assert (total_nll, token_count) == (1.0e16, 3)

    reversed_total_nll = 0.0
    reversed_token_count = 0
    for row in reversed(rows):
        reversed_total_nll, reversed_token_count = module._accumulate_ppl(
            reversed_total_nll, reversed_token_count, row
        )

    assert reversed_token_count == token_count
    assert reversed_total_nll != total_nll


def test_incremental_ppl_finalization_matches_conditional_ppl_exactly() -> None:
    module = pipeline()
    rows = [
        {"selected_logprobs": [-0.125, -0.25]},
        {"selected_logprobs": []},
        {"selected_logprobs": ["-0.375"]},
    ]

    total_nll = 0.0
    token_count = 0
    for row in rows:
        total_nll, token_count = module._accumulate_ppl(total_nll, token_count, row)

    assert module._finalize_ppl(total_nll, token_count) == module.conditional_ppl(rows)


def test_incremental_ppl_preserves_empty_and_invalid_coverage_errors() -> None:
    module = pipeline()

    total_nll, token_count = module._accumulate_ppl(0.0, 0, {"selected_logprobs": []})
    assert (total_nll, token_count) == (0.0, 0)
    with pytest.raises(
        ValueError, match="conditional PPL requires at least one selected token"
    ):
        module._finalize_ppl(total_nll, token_count)
    with pytest.raises(
        ValueError, match="conditional PPL requires at least one selected token"
    ):
        module.conditional_ppl([{"selected_logprobs": []}])

    with pytest.raises(ValueError, match="logprobs must be finite real numbers"):
        module._accumulate_ppl(0.0, 0, {"selected_logprobs": [None]})
    with pytest.raises(ValueError, match="logprobs must be finite real numbers"):
        module.conditional_ppl([{"selected_logprobs": [None]}])


def test_incremental_ppl_finalization_preserves_finite_and_overflow_behavior() -> None:
    module = pipeline()

    assert module._finalize_ppl(0.0, 1) == 1.0

    large_negative = -1.7976931348623157e308
    total_nll, token_count = module._accumulate_ppl(
        0.0, 0, {"selected_logprobs": [large_negative, large_negative]}
    )
    with pytest.raises(ValueError, match="conditional PPL is nonfinite"):
        module._finalize_ppl(total_nll, token_count)
    with pytest.raises(ValueError, match="conditional PPL is nonfinite"):
        module.conditional_ppl(
            [{"selected_logprobs": [large_negative, large_negative]}]
        )
