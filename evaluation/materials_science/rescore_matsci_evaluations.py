#!/usr/bin/env python3
"""Reparse and rescore saved materials-discovery evaluation responses.

This command never calls a model.  It reads the raw candidate text already in
``predictions.jsonl``, applies the current parser from
``evaluate_matsci_vllm.py``, and reports the resulting metric changes for each
``metrics.json``. Pass ``--overwrite`` to atomically update both predictions
and metrics in place so their parser-derived fields remain consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluate_matsci_vllm import (
    atomic_write_json,
    load_rows,
    score_response,
    summarize_scored,
    validate_rows,
)


REPORTED_METRICS = (
    "parse_success",
    "chemical_system_match",
    "exact_formula",
    "format_compliant",
    "geometry_complete",
    "cell_within_1pct_1deg",
    "cell_within_5pct_5deg",
    "exact_formula_and_cell_within_1pct_1deg",
    "exact_formula_and_cell_within_5pct_5deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse saved candidate text and recompute materials evaluation metrics.",
    )
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="Evaluation directory, predictions.jsonl, or a root containing evaluation directories.",
    )
    parser.add_argument(
        "--overwrite", "--write", dest="overwrite", action="store_true",
        help=(
            "Atomically replace every discovered metrics.json and its "
            "predictions.jsonl. The default is a dry run. --write is retained "
            "as a compatibility alias."
        ),
    )
    return parser.parse_args()


def discover_evaluation_dirs(paths: Iterable[Path]) -> List[Path]:
    found: Dict[Path, None] = {}
    for original in paths:
        path = original.resolve()
        if path.is_file():
            if path.name != "predictions.jsonl":
                raise ValueError(f"expected predictions.jsonl, received {path}")
            found[path.parent] = None
        elif (path / "predictions.jsonl").is_file():
            found[path] = None
        elif path.is_dir():
            for prediction_file in sorted(path.rglob("predictions.jsonl")):
                found[prediction_file.parent] = None
        else:
            raise FileNotFoundError(path)
    if not found:
        raise ValueError("no predictions.jsonl files found")
    return list(found)


def read_records_in_file_order(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["row_index"])
            except Exception as exc:
                raise ValueError(f"malformed {path}:{line_number}: {exc}") from exc
            if index in seen:
                raise ValueError(f"duplicate row_index {index} in {path}")
            seen.add(index)
            records.append(record)
    return records


def atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def metric_entry(summary: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    entry = (summary.get("top1_and_pass_at_k") or {}).get(name)
    return entry if isinstance(entry, Mapping) else None


def format_metric_change(
    name: str,
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
) -> str | None:
    before = metric_entry(before_summary, name)
    after = metric_entry(after_summary, name)
    if before is None or after is None:
        return None
    before_count, before_eligible = int(before["count"]), int(before["eligible"])
    after_count, after_eligible = int(after["count"]), int(after["eligible"])
    before_rate, after_rate = float(before["rate"]), float(after["rate"])
    return (
        f"  {name}: {before_count}/{before_eligible} ({before_rate:.4%}) -> "
        f"{after_count}/{after_eligible} ({after_rate:.4%}); "
        f"delta={after_rate - before_rate:+.4%}"
    )


def print_score_changes(
    metrics_path: Path,
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
    overwritten: bool,
) -> None:
    print(f"[{('UPDATED' if overwritten else 'DRY RUN')}] {metrics_path}")
    changes = [
        formatted
        for name in REPORTED_METRICS
        if (formatted := format_metric_change(name, before_summary, after_summary))
        is not None
    ]
    if changes:
        for change in changes:
            print(change)
    else:
        print("  no comparable headline scores found")


def rescore_directory(directory: Path, overwrite: bool) -> None:
    predictions_path = directory / "predictions.jsonl"
    config_path = directory / "run_config.json"
    metrics_path = directory / "metrics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_config = config.get("input") or {}
    input_path = Path(str(input_config["path"]))
    start = int(input_config.get("selected_start_index") or 0)
    selected = int(input_config["selected_rows"])
    rows, _ = load_rows(input_path, start, selected)
    validation = validate_rows(rows)
    rows_by_index = {int(row["row_index"]): row for row in rows}
    records = read_records_in_file_order(predictions_path)
    records_by_index = {int(record["row_index"]): record for record in records}
    if set(records_by_index) != set(rows_by_index):
        missing = sorted(set(rows_by_index) - set(records_by_index))[:5]
        extra = sorted(set(records_by_index) - set(rows_by_index))[:5]
        raise ValueError(f"row selection mismatch for {directory}: missing={missing}, extra={extra}")

    old_scored = []
    new_scored = []
    for row in rows:
        record = records_by_index[int(row["row_index"])]
        if str(record.get("id")) != str(row["id"]):
            raise ValueError(f"id mismatch at row_index {row['row_index']} in {directory}")
        candidates = record.get("candidates") or []
        if not candidates:
            raise ValueError(f"no candidates at row_index {row['row_index']} in {directory}")
        old_scores = [dict(candidate.get("scores") or {}) for candidate in candidates]
        scores = []
        for candidate in candidates:
            score = score_response(str(candidate.get("text") or ""), row)
            candidate["parsed_formula"] = score["predicted_formula"]
            candidate["parsed_geometry"] = score["parsed_geometry"]
            candidate["scores"] = score
            scores.append(score)
        old_scored.append((row, old_scores[0], old_scores))
        new_scored.append((row, scores[0], scores))

    recomputed_old_summary = summarize_scored(old_scored)
    new_summary = summarize_scored(new_scored)
    total = len(rows)
    stored_metrics: Dict[str, Any] = {}
    if metrics_path.is_file():
        stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stored_summary = stored_metrics.get("metrics")
    before_summary = (
        stored_summary if isinstance(stored_summary, Mapping)
        else recomputed_old_summary
    )
    if not overwrite:
        print_score_changes(metrics_path, before_summary, new_summary, False)
        return
    metrics = stored_metrics
    metrics.update({
        "status": "complete",
        "completed_examples": total,
        "selected_examples": total,
        "dataset_profile": validation,
        "metrics": new_summary,
        "rescoring": {
            "script": Path(__file__).name,
            "unix_time": time.time(),
        },
        "updated_unix_time": time.time(),
    })
    atomic_write_jsonl(predictions_path, records)
    atomic_write_json(metrics_path, metrics)
    print_score_changes(metrics_path, before_summary, new_summary, True)


def rescore_paths(paths: Iterable[Path], overwrite: bool) -> int:
    directories = discover_evaluation_dirs(paths)
    failures = 0
    completed = 0
    for directory in directories:
        try:
            rescore_directory(directory, overwrite)
            completed += 1
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {directory / 'metrics.json'}: {exc}", file=sys.stderr)
    action = "updated" if overwrite else "checked"
    print(
        f"Summary: {action}={completed}, errors={failures}, "
        f"discovered={len(directories)}"
    )
    return failures


def main() -> None:
    args = parse_args()
    if rescore_paths(args.paths, args.overwrite):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
