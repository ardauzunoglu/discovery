#!/usr/bin/env python3
"""Extract candidate formulas from materials-evaluation predictions JSONL.

By default, one compact JSON object is written for every candidate whose
formula can be parsed.  The object retains the input row identifiers and the
candidate rank, for example::

    {"row_index":0,"id":"...","material_id":"mp-...","rank":0,"formula":"Li2ScSnYb"}

Raw response text is normalized with the same rules used by the evaluator.
When a response contains multiple ``Candidate:`` lines (for example, one in a
reasoning trace and one in the final answer), the last parseable line wins.
The stored ``parsed_formula`` and ``scores.predicted_formula`` fields are
fallbacks for older files whose response text cannot be parsed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import pyarrow.parquet as pq

from formulas import canonical_formula
from evaluate_matsci_vllm import normalize_response_markup, parse_response


DIRECT_LATEX_SUBSCRIPT_RE = re.compile(r"_(?=\d|\.\d)")
LATEX_FORMULA_WRAPPER_RE = re.compile(
    r"\\(?:boxed|ce|mathbf|mathrm|text)\{([^{}]+)\}", re.IGNORECASE
)
LATEX_STYLE_RE = re.compile(r"\\(?:bf|rm)\b\s*", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
PARENTHESIZED_RE = re.compile(r"\(([^()]*)\)")
TOKEN_RE = re.compile(r"[^\s,/]+")
EXTRACTION_CANDIDATE_LABEL_RE = re.compile(
    r"(?im)(?:\*{1,2})?Candidate(?:\*{1,2})?"
    r"(?:\s*:\s*(?:\*{1,2})?|[^\S\r\n]*\r?\n|(?<=\*)[^\S\r\n]+)"
)
DEFAULT_SPLITS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "materials_science" / "data"
)
CATEGORY_CORRECT = "correct_formula"
CATEGORY_INCORRECT_KNOWN = "incorrect_formula_in_train_or_test"
CATEGORY_INCORRECT_UNKNOWN = "incorrect_formula_not_in_train_or_test"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract predicted material formulas from predictions JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input predictions.jsonl file")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help=(
            "Dataset root; every train/test Parquet file below it defines the "
            "set of known formulas"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file; omit to write to standard output",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "text"),
        default="jsonl",
        help="JSONL records with provenance, or one formula per text line",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Emit candidates with no parseable formula as formula=null (JSONL only)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any candidate has no parseable formula",
    )
    args = parser.parse_args(argv)
    if args.include_missing and args.format != "jsonl":
        parser.error("--include-missing requires --format jsonl")
    if args.output is not None and args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    return args


def iter_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield line_number, record


def discover_split_files(root: Path) -> list[Path]:
    """Find every train/test Parquet file under a dataset root."""
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")

    files = sorted(
        path
        for name in ("train.parquet", "test.parquet")
        for path in root.rglob(name)
    )
    if not files:
        raise FileNotFoundError(f"no train.parquet or test.parquet found under {root}")
    return files


def iter_split_answers(path: Path) -> Iterator[tuple[str, str]]:
    parquet_file = pq.ParquetFile(path)
    required = {"id", "reference_answer_json"}
    missing = sorted(required - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    row_number = 0
    for batch in parquet_file.iter_batches(
        columns=["id", "reference_answer_json"], batch_size=8192
    ):
        data = batch.to_pydict()
        for row_id, encoded_answer in zip(
            data["id"], data["reference_answer_json"], strict=True
        ):
            row_number += 1
            try:
                answer = (
                    json.loads(encoded_answer)
                    if isinstance(encoded_answer, str)
                    else dict(encoded_answer)
                )
                formula = str(answer["candidate_formula"])
            except Exception as exc:
                raise ValueError(
                    f"invalid reference answer at {path} row {row_number}: {exc}"
                ) from exc
            yield str(row_id), formula


def load_split_formula_index(
    root: Path,
) -> tuple[set[str], dict[str, tuple[str, str]], list[Path]]:
    """Load reduced formula membership and gold answers from train plus test."""
    formula_keys: set[str] = set()
    gold_by_id: dict[str, tuple[str, str]] = {}
    files: list[Path] = []
    for path in discover_split_files(root):
        files.append(path)
        for row_id, raw_formula in iter_split_answers(path):
            try:
                formula_key, formula, _ = canonical_formula(raw_formula)
            except Exception as exc:
                raise ValueError(
                    f"invalid formula {raw_formula!r} for {row_id!r} in {path}: {exc}"
                ) from exc
            formula_keys.add(formula_key)
            previous = gold_by_id.get(row_id)
            current = (formula_key, formula)
            if previous is not None and previous != current:
                raise ValueError(
                    f"conflicting gold formulas for id {row_id!r}: "
                    f"{previous[1]} and {formula}"
                )
            gold_by_id[row_id] = current
    return formula_keys, gold_by_id, files


def normalize_formula(value: Any, minimum_elements: int = 1) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        _, formula, chemical_system = canonical_formula(value)
        if len(chemical_system.split("-")) < minimum_elements:
            return None
        return formula
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def normalize_extraction_markup(value: str) -> str:
    """Extend evaluator markup normalization with lossless Unicode/TeX forms."""
    value = unicodedata.normalize("NFKC", value).replace(r"\_", "_")
    return normalize_response_markup(value)


def formula_from_candidate_value(value: str) -> str | None:
    """Parse a formula at the start of a Candidate field's value."""
    value = normalize_extraction_markup(value)
    value = HTML_TAG_RE.sub(" ", value).replace("․", ".")
    # The evaluator handles ``_{2}``; model outputs also commonly use ``_2``.
    value = DIRECT_LATEX_SUBSCRIPT_RE.sub("", value)
    previous = None
    while value != previous:
        previous = value
        value = LATEX_FORMULA_WRAPPER_RE.sub(r"\1", value)
    value = LATEX_STYLE_RE.sub("", value)
    value = value.replace(r"\[", " ").replace(r"\]", " ").replace("$", "")
    value = value.strip().strip("`*_# ")
    tokens = value.split()
    token = tokens[0].rstrip(".,;:`*_# ") if tokens else ""
    if (formula := normalize_formula(token, minimum_elements=2)) is not None:
        return formula

    # NFKC converts Unicode spacing characters to ordinary spaces. Prefer a
    # valid multi-element concatenation for outputs such as ``Dy Ir Nb Tm``
    # before accepting the leading token as a unary formula.
    leading_tokens = [item.strip("()[]{}<>.,;:`*_# ") for item in tokens[:12]]
    for stop in range(len(leading_tokens), 1, -1):
        joined = "".join(leading_tokens[:stop])
        if (formula := normalize_formula(joined, minimum_elements=2)) is not None:
            return formula

    # Occasionally a prose name precedes the actual formula, as in
    # ``BaAlYSelenide (BaAlYSe4)``. Search subsequent delimited tokens while
    # requiring at least two elements to avoid treating ordinary words such
    # as "In" as formulas.
    search_values = PARENTHESIZED_RE.findall(value)
    search_values.extend(match.group() for match in TOKEN_RE.finditer(value))
    for possible_formula in search_values:
        possible_formula = possible_formula.strip("()[]{}<>.,;:`*_# ")
        if (formula := normalize_formula(possible_formula, minimum_elements=2)) is not None:
            return formula
    return normalize_formula(token)


def formula_from_text(text: str) -> str | None:
    """Extract the final predicted formula from raw generated text."""
    normalized = normalize_extraction_markup(text)
    matches = list(EXTRACTION_CANDIDATE_LABEL_RE.finditer(normalized))
    for index in range(len(matches) - 1, -1, -1):
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        # A formula and a short parenthetical explanation fit comfortably in
        # this bound, which prevents later free-form prose from being searched.
        candidate_value = normalized[start:min(end, start + 500)]
        if (formula := formula_from_candidate_value(candidate_value)) is not None:
            return formula

    # Preserve the evaluator's GPT-OSS final-channel and bare-line fallbacks
    # for responses that contain no usable Candidate field.
    return normalize_formula(parse_response(text).get("formula"))


def extract_formula(candidate: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return a reduced formula and the field from which it was extracted."""
    text = candidate.get("text")
    if text is not None:
        if (formula := formula_from_text(str(text))) is not None:
            return formula, "text"

    if (formula := normalize_formula(candidate.get("parsed_formula"))) is not None:
        return formula, "parsed_formula"

    scores = candidate.get("scores")
    if isinstance(scores, Mapping):
        if (formula := normalize_formula(scores.get("predicted_formula"))) is not None:
            return formula, "scores.predicted_formula"

    return None, None


def gold_formula_for_record(
    record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, tuple[str, str]],
) -> tuple[str, str]:
    row_id = str(record.get("id") or "")
    if row_id in gold_by_id:
        return gold_by_id[row_id]

    # This supports compatible prediction files whose selected questions are
    # no longer present in the supplied split version.
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        scores = candidate.get("scores")
        if not isinstance(scores, Mapping):
            continue
        raw_formula = scores.get("gold_formula")
        if raw_formula:
            formula_key, formula, _ = canonical_formula(raw_formula)
            return formula_key, formula
    raise ValueError(
        f"cannot determine the correct formula for prediction id {row_id!r}"
    )


def categorize_formula(
    formula: str, gold_formula_key: str, split_formula_keys: set[str]
) -> str:
    formula_key, _, _ = canonical_formula(formula)
    if formula_key == gold_formula_key:
        return CATEGORY_CORRECT
    if formula_key in split_formula_keys:
        return CATEGORY_INCORRECT_KNOWN
    return CATEGORY_INCORRECT_UNKNOWN


def output_record(
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_index: int,
    formula: str | None,
    source: str | None,
    gold_formula: str,
    category: str | None,
) -> dict[str, Any]:
    return {
        "row_index": record.get("row_index"),
        "id": record.get("id"),
        "material_id": record.get("material_id"),
        "rank": candidate.get("rank", candidate_index),
        "formula": formula,
        "source": source,
        "gold_formula": gold_formula,
        "category": category,
    }


def write_extractions(
    args: argparse.Namespace,
    destination: TextIO,
    split_formula_keys: set[str],
    gold_by_id: Mapping[str, tuple[str, str]],
) -> tuple[int, int, int, Counter[str]]:
    row_count = candidate_count = missing_count = 0
    category_counts: Counter[str] = Counter()
    for line_number, record in iter_records(args.input):
        row_count += 1
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(
                f"expected a candidates list at {args.input}:{line_number}"
            )
        gold_formula_key, gold_formula = gold_formula_for_record(
            record, candidates, gold_by_id
        )
        for candidate_index, candidate in enumerate(candidates):
            candidate_count += 1
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"candidate {candidate_index} is not an object at "
                    f"{args.input}:{line_number}"
                )
            formula, source = extract_formula(candidate)
            category = None
            if formula is None:
                missing_count += 1
                if not args.include_missing:
                    continue
            else:
                category = categorize_formula(
                    formula, gold_formula_key, split_formula_keys
                )
                category_counts[category] += 1
            if args.format == "text":
                destination.write(formula + "\n")
            else:
                extracted = output_record(
                    record,
                    candidate,
                    candidate_index,
                    formula,
                    source,
                    gold_formula,
                    category,
                )
                destination.write(
                    json.dumps(extracted, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
    return row_count, candidate_count, missing_count, category_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_formula_keys, gold_by_id, split_files = load_split_formula_index(
        args.splits_dir
    )
    output_context = (
        args.output.open("w", encoding="utf-8")
        if args.output is not None
        else nullcontext(sys.stdout)
    )
    with output_context as destination:
        rows, candidates, missing, category_counts = write_extractions(
            args, destination, split_formula_keys, gold_by_id
        )

    extracted = candidates - missing
    print(
        f"Read {rows} rows and {candidates} candidates; extracted "
        f"{extracted} formulas; {missing} were unparseable.",
        file=sys.stderr,
    )
    print(
        f"Checked against {len(split_formula_keys)} formulas from "
        f"{len(split_files)} split files: "
        f"{CATEGORY_CORRECT}={category_counts[CATEGORY_CORRECT]}, "
        f"{CATEGORY_INCORRECT_KNOWN}={category_counts[CATEGORY_INCORRECT_KNOWN]}, "
        f"{CATEGORY_INCORRECT_UNKNOWN}={category_counts[CATEGORY_INCORRECT_UNKNOWN]}.",
        file=sys.stderr,
    )
    if args.strict and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
