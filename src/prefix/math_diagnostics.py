"""Conservative, deterministic answer diagnostics for Math-500 responses."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Literal, TypeAlias


ExtractionStatus: TypeAlias = Literal["answer", "null", "ambiguous", "indeterminate"]
ComparisonVerdict: TypeAlias = Literal[
    "equivalent", "different", "ambiguous", "null", "indeterminate"
]


@dataclass(frozen=True)
class AnswerExtraction:
    """One conservatively extracted candidate and its provenance."""

    status: ExtractionStatus
    value: str | None
    raw_candidate: str | None
    method: str | None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerComparison:
    """A side-by-side comparison retaining both extraction records."""

    verdict: ComparisonVerdict
    left: AnswerExtraction
    right: AnswerExtraction
    reason: str


_ANSWER_LINE_RE = re.compile(
    r"\bThe answer is\s*(?:\(\s*)?(?P<candidate>.+?)" + r"(?:\s*\))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DECIMAL_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_RATIONAL_RE = re.compile(r"([+-]?\d+)\s*/\s*([+-]?\d+)")
_FRAC_RE = re.compile(r"\\frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")


def _boxed_candidates(response: str) -> list[str]:
    candidates: list[str] = []
    marker = r"\boxed{"
    start = 0
    while (opening := response.find(marker, start)) >= 0:
        body_start = opening + len(marker)
        depth = 1
        index = body_start
        while index < len(response) and depth:
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            candidate = response[body_start : index - 1]
            if candidate.strip():
                candidates.append(candidate)
            start = index
        else:
            break
    return candidates


def _normalized(candidate: str) -> str:
    return re.sub(r"\s+", "", candidate).strip()


def _number(candidate: str) -> Fraction | None:
    normalized = _normalized(candidate)
    if match := _FRAC_RE.fullmatch(normalized):
        numerator, denominator = match.groups()
        try:
            return Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None
    if match := _RATIONAL_RE.fullmatch(normalized):
        try:
            return Fraction(int(match.group(1)), int(match.group(2)))
        except ZeroDivisionError:
            return None
    if _DECIMAL_RE.fullmatch(normalized):
        try:
            return Fraction(Decimal(normalized))
        except (InvalidOperation, ValueError):
            return None
    return None


def _unsupported(candidate: str) -> bool:
    normalized = _normalized(candidate)
    if (normalized.startswith("[") and normalized.endswith("]")) or (
        normalized.startswith("{") and normalized.endswith("}")
    ):
        return True
    return "\\" in normalized and not _FRAC_RE.fullmatch(normalized)


def _candidate_equivalent(left: str, right: str) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    if left_number is None and right_number is None:
        return (
            not _unsupported(left)
            and not _unsupported(right)
            and _normalized(left) == _normalized(right)
        )
    return False


def _answer_extraction(
    candidate: str, method: str, candidates: tuple[str, ...]
) -> AnswerExtraction:
    value = candidate.strip()
    status: ExtractionStatus = "indeterminate" if _unsupported(value) else "answer"
    return AnswerExtraction(status, value, candidate, method, candidates)


def extract_final_answer(response: str) -> AnswerExtraction:
    """Extract the last valid boxed or explicit final answer conservatively."""

    boxed = _boxed_candidates(response)
    explicit = [
        match.group("candidate").strip().rstrip(".!?").strip()
        for match in _ANSWER_LINE_RE.finditer(response)
    ]
    if boxed and explicit:
        boxed_value = boxed[-1].strip()
        explicit_value = explicit[-1].strip()
        candidates = (boxed_value, explicit_value)
        if not _candidate_equivalent(boxed_value, explicit_value):
            return AnswerExtraction("ambiguous", None, None, None, candidates)
        return _answer_extraction(explicit[-1], "answer_sentence", candidates)
    if boxed:
        return _answer_extraction(
            boxed[-1], "boxed", tuple(item.strip() for item in boxed)
        )
    if explicit:
        return _answer_extraction(
            explicit[-1], "answer_sentence", tuple(item.strip() for item in explicit)
        )
    return AnswerExtraction("null", None, None, None)


AnswerInput: TypeAlias = str | AnswerExtraction


def _as_extraction(candidate: AnswerInput) -> AnswerExtraction:
    if isinstance(candidate, AnswerExtraction):
        return candidate
    return AnswerExtraction("answer", candidate, candidate, "candidate", (candidate,))


def compare_math_answers(left: AnswerInput, right: AnswerInput) -> AnswerComparison:
    """Compare supported candidates, preserving indeterminate cases."""

    left_result = _as_extraction(left)
    right_result = _as_extraction(right)
    if left_result.status == "ambiguous" or right_result.status == "ambiguous":
        return AnswerComparison(
            "ambiguous", left_result, right_result, "ambiguous_input"
        )
    if left_result.status == "null" or right_result.status == "null":
        return AnswerComparison("null", left_result, right_result, "missing_answer")
    if left_result.status == "indeterminate" or right_result.status == "indeterminate":
        return AnswerComparison(
            "indeterminate", left_result, right_result, "unsupported_expression"
        )
    assert left_result.value is not None and right_result.value is not None
    if _unsupported(left_result.value) or _unsupported(right_result.value):
        return AnswerComparison(
            "indeterminate", left_result, right_result, "unsupported_expression"
        )
    left_number = _number(left_result.value)
    right_number = _number(right_result.value)
    if left_number is not None and right_number is not None:
        reason = (
            "rational"
            if "/" in left_result.value
            or "/" in right_result.value
            or "\\frac" in left_result.value
            or "\\frac" in right_result.value
            else "numeric"
        )
        verdict: ComparisonVerdict = (
            "equivalent" if left_number == right_number else "different"
        )
        return AnswerComparison(verdict, left_result, right_result, reason)
    if left_number is None and right_number is None:
        if _normalized(left_result.value) == _normalized(right_result.value):
            return AnswerComparison(
                "equivalent", left_result, right_result, "normalized_string"
            )
        return AnswerComparison(
            "indeterminate", left_result, right_result, "unsupported_expression"
        )
    return AnswerComparison(
        "indeterminate", left_result, right_result, "unsupported_expression"
    )


__all__ = [
    "AnswerComparison",
    "AnswerExtraction",
    "compare_math_answers",
    "extract_final_answer",
]
