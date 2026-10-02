"""Failing-first contract tests for deterministic Math-500 diagnostics.

These tests intentionally specify the diagnostic API without implementing it.
The API is a local comparison aid and must not replace the Gemini judge.
"""

from __future__ import annotations

from typing import Literal

import pytest

from prefix.math_diagnostics import (
    AnswerComparison,
    AnswerExtraction,
    compare_math_answers,
    extract_final_answer,
)


ExtractionStatus = Literal["answer", "null", "ambiguous", "indeterminate"]
ComparisonVerdict = Literal[
    "equivalent", "different", "ambiguous", "null", "indeterminate"
]


def test_extracts_last_valid_boxed_answer_and_preserves_candidate_metadata() -> None:
    response = r"""
    First attempt: \boxed{}
    Revision: \boxed{\frac{-6}{8}}
    Final check: \boxed{ 3/4 }
    """

    result: AnswerExtraction = extract_final_answer(response)

    assert result.status == "answer"
    assert result.value == "3/4"
    assert result.raw_candidate == " 3/4 "
    assert result.method == "boxed"


def test_extracts_last_explicit_final_answer_sentence() -> None:
    response = (
        "The answer is 11 while checking the arithmetic.\n"
        "Therefore, the final answer is -0.75.\n"
        "The answer is -0.75."
    )

    result: AnswerExtraction = extract_final_answer(response)

    assert result.status == "answer"
    assert result.value == "-0.75"
    assert result.raw_candidate == "-0.75"
    assert result.method == "answer_sentence"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("0.75", r"\frac{3}{4}"),
        ("-6/8", "-0.75"),
        ("+2", "2.0"),
        ("  x^2 + 1  ", "x^2+1"),
    ],
)
def test_comparison_reports_numeric_rational_and_normalized_string_equivalence(
    left: str, right: str
) -> None:
    result: AnswerComparison = compare_math_answers(left, right)

    assert result.verdict == "equivalent"
    assert result.left.raw_candidate == left
    assert result.right.raw_candidate == right
    assert result.reason in {"numeric", "rational", "normalized_string"}


def test_comparison_distinguishes_safe_numeric_inequality() -> None:
    result: AnswerComparison = compare_math_answers("-1/2", "1/2")

    assert result.verdict == "different"
    assert result.reason in {"numeric", "rational"}


def test_conflicting_boxed_and_answer_sentence_is_ambiguous() -> None:
    response = r"The derivation suggests \boxed{2}, but The answer is 3."

    result: AnswerExtraction = extract_final_answer(response)

    assert result.status == "ambiguous"
    assert result.value is None
    assert result.raw_candidate is None
    assert result.method is None
    assert result.candidates == ("2", "3")


def test_missing_answer_is_null_not_false() -> None:
    result: AnswerExtraction = extract_final_answer(
        "The response contains reasoning but no final candidate."
    )

    assert result.status == "null"
    assert result.value is None
    assert result.raw_candidate is None
    assert result.method is None


@pytest.mark.parametrize(
    "candidate",
    [r"\sqrt{2}", r"\{1,2,3\}", r"[0,1]", r"\int_0^1 x dx"],
)
def test_unsupported_symbolic_set_or_interval_comparison_is_indeterminate(
    candidate: str,
) -> None:
    result: AnswerComparison = compare_math_answers(candidate, "0.5")

    assert result.verdict == "indeterminate"
    assert result.reason == "unsupported_expression"
    assert result.left.raw_candidate == candidate
    assert result.right.raw_candidate == "0.5"


def test_comparison_result_is_side_by_side_and_preserves_extraction_metadata() -> None:
    left_response = r"Work... \boxed{\frac{2}{4}}"
    right_response = "Work... The answer is 0.50."

    result: AnswerComparison = compare_math_answers(
        extract_final_answer(left_response), extract_final_answer(right_response)
    )

    assert result.verdict == "equivalent"
    assert result.left == AnswerExtraction(
        status="answer",
        value=r"\frac{2}{4}",
        raw_candidate=r"\frac{2}{4}",
        method="boxed",
        candidates=(r"\frac{2}{4}",),
    )
    assert result.right.method == "answer_sentence"
    assert result.right.raw_candidate == "0.50"
