#!/usr/bin/env python3
"""Calculate recorded token cost and token distributions for a distillation JSONL file."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


TOKENS_PER_MILLION = Decimal("1000000")


def nonnegative_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="distillation JSONL file")
    parser.add_argument(
        "--input-price",
        type=nonnegative_decimal,
        default=Decimal("5"),
        metavar="DOLLARS_PER_MILLION",
        help="input-token price (default: $5 per 1M tokens)",
    )
    parser.add_argument(
        "--output-price",
        type=nonnegative_decimal,
        default=Decimal("25"),
        metavar="DOLLARS_PER_MILLION",
        help="output-token price (default: $25 per 1M tokens)",
    )
    return parser.parse_args()


def token_count(row: dict[str, Any], field: str, line_number: int) -> int:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError(f"line {line_number}: missing teacher object")
    value = teacher.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"line {line_number}: teacher.{field} must be a nonnegative integer"
        )
    return value


def percentile(values: list[int], p: float) -> float:
    """Linearly interpolated percentile."""
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def print_distribution(name: str, values: list[int]) -> None:
    print(f"\n{name}:")
    print(f"  Mean:   {statistics.mean(values):,.1f}")
    print(f"  Min:    {min(values):,}")
    print(f"  P25:    {percentile(values, 0.25):,.1f}")
    print(f"  Median: {statistics.median(values):,.1f}")
    print(f"  P75:    {percentile(values, 0.75):,.1f}")
    print(f"  P90:    {percentile(values, 0.90):,.1f}")
    print(f"  P95:    {percentile(values, 0.95):,.1f}")
    print(f"  Max:    {max(values):,}")


def main() -> int:
    args = parse_args()
    try:
        raw_lines = args.input.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2

    # A generation process may be appending the final row while this script reads.
    if raw_lines and not raw_lines[-1].endswith((b"\n", b"\r")):
        print("warning: ignoring an incomplete final line", file=sys.stderr)
        raw_lines.pop()

    examples = 0
    input_tokens = 0
    output_tokens = 0

    input_token_values: list[int] = []
    reasoning_token_values: list[int] = []
    answer_token_values: list[int] = []

    try:
        for line_number, raw in enumerate(raw_lines, 1):
            if not raw.strip():
                raise ValueError(f"line {line_number}: blank line")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row is not a JSON object")

            row_input_tokens = token_count(row, "input_tokens", line_number)
            row_output_tokens = token_count(row, "output_tokens", line_number)
            row_reasoning_tokens = token_count(row, "reasoning_tokens", line_number)

            if row_reasoning_tokens > row_output_tokens:
                raise ValueError(
                    f"line {line_number}: reasoning_tokens exceeds output_tokens"
                )

            input_tokens += row_input_tokens
            output_tokens += row_output_tokens

            input_token_values.append(row_input_tokens)
            reasoning_token_values.append(row_reasoning_tokens)
            answer_token_values.append(row_output_tokens - row_reasoning_tokens)

            examples += 1

    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    input_cost = Decimal(input_tokens) * args.input_price / TOKENS_PER_MILLION
    output_cost = Decimal(output_tokens) * args.output_price / TOKENS_PER_MILLION
    total_cost = input_cost + output_cost
    average_cost = total_cost / examples if examples else Decimal(0)

    print(f"File: {args.input.resolve()}")
    print(f"Examples: {examples:,}")
    print(f"Input tokens: {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Input cost (@ ${args.input_price}/1M): ${input_cost:,.6f}")
    print(f"Output cost (@ ${args.output_price}/1M): ${output_cost:,.6f}")
    print(f"Total recorded cost: ${total_cost:,.6f}")
    print(f"Average per example: ${average_cost:,.6f}")

    if examples:
        print("\nToken distributions")
        print_distribution("Input tokens", input_token_values)
        print_distribution("Reasoning tokens", reasoning_token_values)
        print_distribution(
            "Output tokens excluding reasoning", answer_token_values
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())