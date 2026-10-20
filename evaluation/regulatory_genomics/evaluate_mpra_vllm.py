#!/usr/bin/env python3
"""Run and score vLLM inference on MPRA structured-answer prediction data.

The evaluator supports canonical genomic-variant records and compact local-edit
records. ``prompt_messages`` contains the messages shown to the model, while the
sequence and genomic columns define the gold answer. The answer format is
normally inferred from Parquet schema metadata. Only the structured final
answer is scored; a free-form reasoning trace is retained but remains unscored.

Example:
    python evaluation/regulatory_genomics/evaluate_mpra_vllm.py \
      --input-parquet data/regulatory_genomics/data/evaluation/test.parquet \
      --model /path/to/checkpoint \
      --output-dir outputs/regulatory_genomics/evaluation/checkpoint

The output directory contains append-only ``predictions.jsonl``, a reproducible
``run_config.json``, and aggregate ``metrics.json``. Interrupted runs resume by
default. vLLM is imported lazily, so ``--help`` and ``--validate-only`` work in
CPU-only data-processing environments.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cot_prompt import COT_PROMPT_VERSION, cot_prompt_messages


REQUIRED_COLUMNS = (
    "pair_id",
    "split",
    "reference_assembly",
    "chromosome",
    "position",
    "reference_sequence",
    "alternate_sequence",
    "reference_allele",
    "alternate_allele",
    "reference_sequence_length",
    "alternate_sequence_length",
    "sequence_length_change",
    "data_project",
    "variant_class",
    "prompt_messages",
)
OPTIONAL_COLUMNS = (
    "source",
    "task",
    "K562_delta_log2FC",
    "HepG2_delta_log2FC",
    "SKNSH_delta_log2FC",
)

REASONING_LABEL_RE = re.compile(r"(?im)^\s*Reasoning\s*:")
VARIANT_LABEL_RE = re.compile(r"(?im)^\s*Variant\s*:\s*")
LOCAL_EDIT_LABEL_RE = re.compile(r"(?im)^\s*Edit\s*:\s*")
VARIANT_RECORD_RE = re.compile(
    r"(?i)(GRCh37)\|(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y|MT)\|"
    r"([1-9][0-9]*)\|([ACGTN]+)\|([ACGTN]+)"
)
LOCAL_EDIT_RECORD_RE = re.compile(
    r"(?i)\b(Substitute|Insert)\s*,\s*([1-9][0-9]*)\s*,\s*([ACGTN]+)\b"
)
GPT_OSS_FINAL_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:(?:<\|start\|>)?assistant(?:<\|channel\|>)?)?final"
    r"(?:<\|message\|>)?\s*:?\s*"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local model with vLLM on MPRA chat parquet data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument(
        "--answer-format",
        choices=("auto", "variant-record", "local-edit"),
        default="auto",
        help="Structured answer to parse and score; auto uses Parquet metadata.",
    )
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model id or local checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="Validate/profile the parquet without loading vLLM.")

    data = parser.add_argument_group("data selection")
    data.add_argument("--start-index", type=int, default=0)
    data.add_argument("--max-examples", type=int, default=None)

    prompt = parser.add_argument_group("prompt fidelity")
    prompt.add_argument(
        "--chat-template-source",
        choices=("parquet", "model"),
        default="parquet",
        help="Use the embedded dataset template or the tokenizer's own template.",
    )
    prompt.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help="Explicit Jinja template; overrides --chat-template-source.",
    )
    prompt.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="GPT-OSS reasoning effort passed to the model chat template.",
    )
    prompt.add_argument(
        "--cot-prompt",
        action="store_true",
        help="Rewrite a supported no-CoT instruction to its exact CoT training format.",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--request-chunk-size", type=int, default=1024, help="Requests submitted per vLLM call/checkpoint.")
    generation.add_argument("--max-tokens", type=int, default=512)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--top-k", type=int, default=-1)
    generation.add_argument("--min-p", type=float, default=0.0)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--presence-penalty", type=float, default=0.0)
    generation.add_argument("--frequency-penalty", type=float, default=0.0)
    generation.add_argument("--n", type=int, default=1, help="Candidates per prompt; summary reports top-1 and pass@k.")
    generation.add_argument("--seed", type=int, default=13)

    engine = parser.add_argument_group("vLLM engine")
    engine.add_argument("--tokenizer", type=str, default=None)
    engine.add_argument("--revision", type=str, default=None)
    engine.add_argument("--tokenizer-revision", type=str, default=None)
    engine.add_argument("--dtype", type=str, default="auto", choices=("auto", "half", "float16", "bfloat16", "float", "float32"))
    engine.add_argument("--quantization", type=str, default=None)
    engine.add_argument("--tensor-parallel-size", type=int, default=1)
    engine.add_argument("--pipeline-parallel-size", type=int, default=1)
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    engine.add_argument("--max-model-len", type=int, default=None)
    engine.add_argument("--max-num-seqs", type=int, default=None)
    engine.add_argument("--cpu-offload-gb", type=float, default=0.0)
    engine.add_argument("--swap-space", type=float, default=4.0)
    engine.add_argument("--download-dir", type=str, default=None)
    engine.add_argument("--trust-remote-code", action="store_true")
    engine.add_argument("--enforce-eager", action="store_true")
    engine.add_argument("--disable-custom-all-reduce", action="store_true")
    engine.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Useful here because every request shares a substantial instruction prefix.",
    )

    output = parser.add_argument_group("output/recovery")
    output.add_argument(
        "--save-every",
        type=int,
        default=1000,
        metavar="K",
        help="Durably save predictions and report current metrics every K completed examples.",
    )
    output.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--quiet-vllm", action="store_true", help="Disable vLLM's per-call progress bar.")

    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be > 0")
    if args.request_chunk_size <= 0 or args.max_tokens <= 0:
        parser.error("--request-chunk-size and --max-tokens must be > 0")
    if args.save_every <= 0:
        parser.error("--save-every must be > 0")
    if args.n <= 0:
        parser.error("--n must be > 0")
    if args.n > 1 and args.temperature <= 0:
        parser.error("--n > 1 requires --temperature > 0 so candidates are not identical")
    if not -2.0 <= args.presence_penalty <= 2.0:
        parser.error("--presence-penalty must be in [-2, 2]")
    if not -2.0 <= args.frequency_penalty <= 2.0:
        parser.error("--frequency-penalty must be in [-2, 2]")
    if not 0.0 <= args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in [0, 1]")
    if not args.validate_only and not args.model:
        parser.error("--model is required unless --validate-only is used")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if args.overwrite and args.resume:
        # Overwrite has an unambiguous meaning and takes precedence over the default.
        args.resume = False
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_chat_template(parquet_file: pq.ParquetFile) -> Optional[str]:
    metadata = parquet_file.schema_arrow.metadata or {}
    value = metadata.get(b"chat_template")
    return value.decode("utf-8") if value else None


def load_rows(path: Path, start_index: int, max_examples: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {path}")
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(REQUIRED_COLUMNS) - available)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {', '.join(missing)}")

    end_index = parquet_file.metadata.num_rows
    if start_index >= end_index:
        raise ValueError(f"--start-index {start_index} is outside a {end_index}-row parquet")
    if max_examples is not None:
        end_index = min(end_index, start_index + max_examples)

    columns = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in available]
    # This dataset is small enough to read selected columns once. vLLM itself still
    # receives bounded chunks, providing both continuous batching and checkpoints.
    table = parquet_file.read(columns=columns).slice(start_index, end_index - start_index)
    rows = table.to_pylist()
    for offset, row in enumerate(rows):
        row["row_index"] = start_index + offset

    profile = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "parquet_rows": parquet_file.metadata.num_rows,
        "selected_start_index": start_index,
        "selected_rows": len(rows),
        "row_groups": parquet_file.metadata.num_row_groups,
        "schema_columns": len(parquet_file.schema_arrow.names),
        "embedded_chat_template": parquet_chat_template(parquet_file) is not None,
        "schema_metadata": {
            key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
            for key, value in (parquet_file.schema_arrow.metadata or {}).items()
            if key != b"chat_template"
        },
    }
    return rows, profile


def resolve_answer_format(requested: str, profile: Mapping[str, Any]) -> str:
    if requested != "auto":
        return requested
    metadata = profile.get("schema_metadata") or {}
    task = str(metadata.get("task") or "")
    completion_format = str(metadata.get("completion_format") or "").lower()
    if task == "goal_conditioned_mpra_local_edit_design" or "<edit type>" in completion_format:
        return "local-edit"
    if "variant record" in completion_format:
        return "variant-record"
    raise ValueError(
        "Could not infer the answer format from Parquet metadata. "
        "Pass --answer-format variant-record or --answer-format local-edit."
    )


def validate_rows(
    rows: Sequence[Mapping[str, Any]], answer_format: str
) -> Dict[str, Any]:
    errors: List[str] = []
    projects: Counter[str] = Counter()
    variant_types: Counter[str] = Counter()
    gold_lengths: Counter[int] = Counter()

    for row in rows:
        idx = int(row["row_index"])
        ref = str(row["reference_sequence"] or "").upper()
        alt = str(row["alternate_sequence"] or "").upper()
        gold_answer = canonical_answer(row, answer_format)
        messages = row["prompt_messages"]
        if not ref or set(ref) - set("ACGTN"):
            errors.append(f"row {idx}: reference_sequence is empty or non-DNA")
        if not alt or set(alt) - set("ACGTN"):
            errors.append(f"row {idx}: alternate_sequence is empty or non-DNA")
        if len(ref) != int(row["reference_sequence_length"]):
            errors.append(f"row {idx}: reference_sequence_length disagrees with sequence")
        if len(alt) != int(row["alternate_sequence_length"]):
            errors.append(f"row {idx}: alternate_sequence_length disagrees with sequence")
        if len(alt) - len(ref) != int(row["sequence_length_change"]):
            errors.append(f"row {idx}: sequence_length_change disagrees with sequence lengths")
        if not isinstance(messages, list) or not messages:
            errors.append(f"row {idx}: prompt_messages is not a nonempty list")
        else:
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                    errors.append(f"row {idx}: malformed prompt_messages entry")
                    break
                if not isinstance(message.get("content"), str):
                    errors.append(f"row {idx}: message content is not a string")
                    break
        projects[str(row.get("data_project") or "unknown")] += 1
        variant_types[variant_type(row)] += 1
        gold_lengths[len(gold_answer)] += 1
        if len(errors) >= 25:
            break

    if errors:
        raise ValueError("Input validation failed:\n  " + "\n  ".join(errors))
    return {
        "validated_rows": len(rows),
        "data_projects": dict(projects),
        "variant_types": dict(variant_types),
        "answer_format": answer_format,
        "gold_answer_lengths": {
            str(key): value for key, value in sorted(gold_lengths.items())
        },
    }


def variant_type(row: Mapping[str, Any]) -> str:
    ref = str(row.get("reference_allele") or "")
    alt = str(row.get("alternate_allele") or "")
    if len(ref) == len(alt) == 1:
        return "snv"
    if len(alt) > len(ref):
        return "insertion"
    if len(alt) < len(ref):
        return "deletion"
    return "multi_base_substitution"


def configure_cot_prompts(rows: Sequence[Dict[str, Any]]) -> None:
    """Apply the same output-instruction rewrite used to build CoT SFT data."""
    for row in rows:
        try:
            row["prompt_messages"] = cot_prompt_messages(row["prompt_messages"])
        except ValueError as exc:
            raise ValueError(f"row {row['row_index']}: {exc}") from exc


def effect_bin(row: Mapping[str, Any]) -> str:
    values = []
    for name in ("K562_delta_log2FC", "HepG2_delta_log2FC", "SKNSH_delta_log2FC"):
        value = row.get(name)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(abs(value))
    maximum = max(values, default=0.0)
    if maximum < 0.1:
        return "<0.1"
    if maximum < 0.5:
        return "0.1-0.5"
    if maximum < 1.0:
        return "0.5-1.0"
    return ">=1.0"


def strip_generation_protocol_prefix(text: str) -> str:
    """Remove a leaked GPT-OSS ``final`` channel prefix."""
    text = text or ""
    match = GPT_OSS_FINAL_PREFIX_RE.match(text)
    return text[match.end() :] if match else text


def canonical_variant(row: Mapping[str, Any]) -> str:
    assembly = str(row.get("reference_assembly") or "").lower()
    if assembly != "grch37":
        raise ValueError(f"expected GRCh37, got {row.get('reference_assembly')!r}")
    chromosome = str(row.get("chromosome") or "").removeprefix("chr").upper()
    position = int(row["position"])
    reference = str(row.get("reference_allele") or "").upper()
    alternate = str(row.get("alternate_allele") or "").upper()
    return f"GRCh37|{chromosome}|{position}|{reference}|{alternate}"


def canonical_local_edit(row: Mapping[str, Any]) -> str:
    """Derive the compact edit using the same convention as dataset creation."""
    reference = str(row.get("reference_allele") or "").upper()
    alternate = str(row.get("alternate_allele") or "").upper()
    reference_sequence = str(row.get("reference_sequence") or "").upper()
    alternate_sequence = str(row.get("alternate_sequence") or "").upper()
    window_length = max(len(reference_sequence), len(alternate_sequence))
    anchor_length = max(len(reference), len(alternate))
    allele_offset = max(0, (window_length - 1 - anchor_length) // 2)
    position = allele_offset + 1

    observed_reference = reference_sequence[allele_offset : allele_offset + len(reference)]
    if observed_reference != reference:
        raise ValueError(
            f"local position {position} contains {observed_reference!r}, "
            f"not reference allele {reference!r}"
        )
    if len(reference) == len(alternate) == 1 and reference != alternate:
        return f"Substitute, {position}, {alternate}"
    if len(reference) == 1 and len(alternate) > 1 and alternate.startswith(reference):
        return f"Insert, {position}, {alternate[len(reference):]}"
    raise ValueError(
        f"unsupported local edit {reference}>{alternate}; expected an SNV or "
        "a left-anchored insertion"
    )


def canonical_answer(row: Mapping[str, Any], answer_format: str) -> str:
    if answer_format == "variant-record":
        return canonical_variant(row)
    if answer_format == "local-edit":
        return canonical_local_edit(row)
    raise ValueError(f"unsupported answer format: {answer_format}")


def extract_variant_record(text: str) -> Optional[str]:
    """Extract and normalize the last canonical GRCh37 variant record."""
    text = strip_generation_protocol_prefix(text)
    matches = list(VARIANT_RECORD_RE.finditer(text))
    if not matches:
        return None
    _, chromosome, position, reference, alternate = matches[-1].groups()
    return f"GRCh37|{chromosome.upper()}|{int(position)}|{reference.upper()}|{alternate.upper()}"


def extract_local_edit(text: str) -> Optional[str]:
    """Extract and normalize the last compact local-edit record."""
    text = strip_generation_protocol_prefix(text)
    matches = list(LOCAL_EDIT_RECORD_RE.finditer(text))
    if not matches:
        return None
    kind, position, sequence = matches[-1].groups()
    normalized_kind = "Substitute" if kind.lower() == "substitute" else "Insert"
    return f"{normalized_kind}, {int(position)}, {sequence.upper()}"


def extract_answer(text: str, answer_format: str) -> Optional[str]:
    if answer_format == "variant-record":
        return extract_variant_record(text)
    if answer_format == "local-edit":
        return extract_local_edit(text)
    raise ValueError(f"unsupported answer format: {answer_format}")


def score_variant(
    predicted: Optional[str], row: Mapping[str, Any], raw_text: Optional[str] = None
) -> Dict[str, Any]:
    gold = canonical_variant(row)
    parsed = bool(predicted)
    predicted_parts = predicted.split("|") if predicted else []
    gold_parts = gold.split("|")
    generated_content = strip_generation_protocol_prefix(raw_text or "")
    bare_variant_only = bool(VARIANT_RECORD_RE.fullmatch(generated_content.strip()))
    labels = list(VARIANT_LABEL_RE.finditer(generated_content))
    labeled_variant_only = bool(
        labels and VARIANT_RECORD_RE.fullmatch(generated_content[labels[-1].end() :].strip())
    )
    component_matches = [
        parsed and predicted_parts[index].upper() == gold_parts[index].upper()
        for index in range(5)
    ]
    coordinate_error = None
    if parsed and component_matches[0] and component_matches[1]:
        coordinate_error = abs(int(predicted_parts[2]) - int(gold_parts[2]))

    return {
        "has_reasoning_label": bool(REASONING_LABEL_RE.search(raw_text or "")),
        "has_variant_label": bool(labels),
        "bare_variant_only": bare_variant_only,
        "format_compliant": bare_variant_only or labeled_variant_only,
        "parse_success": parsed,
        "exact_match": parsed and predicted == gold,
        "assembly_exact": component_matches[0],
        "chromosome_exact": component_matches[1],
        "position_exact": component_matches[2],
        "reference_allele_exact": component_matches[3],
        "alternate_allele_exact": component_matches[4],
        "locus_exact": all(component_matches[:3]),
        "edit_exact": all(component_matches[3:]),
        "coordinate_error": coordinate_error,
    }


def score_local_edit(
    predicted: Optional[str], row: Mapping[str, Any], raw_text: Optional[str] = None
) -> Dict[str, Any]:
    gold = canonical_local_edit(row)
    predicted_match = LOCAL_EDIT_RECORD_RE.fullmatch(predicted or "")
    parsed = predicted_match is not None
    gold_match = LOCAL_EDIT_RECORD_RE.fullmatch(gold)
    assert gold_match is not None
    predicted_parts = predicted_match.groups() if predicted_match else ("", "", "")
    gold_parts = gold_match.groups()
    generated_content = strip_generation_protocol_prefix(raw_text or "")
    bare_edit_only = bool(LOCAL_EDIT_RECORD_RE.fullmatch(generated_content.strip()))
    labels = list(LOCAL_EDIT_LABEL_RE.finditer(generated_content))
    labeled_edit_only = bool(
        labels
        and LOCAL_EDIT_RECORD_RE.fullmatch(generated_content[labels[-1].end() :].strip())
    )
    component_matches = [
        parsed and predicted_parts[0].lower() == gold_parts[0].lower(),
        parsed and int(predicted_parts[1]) == int(gold_parts[1]),
        parsed and predicted_parts[2].upper() == gold_parts[2].upper(),
    ]
    position_error = (
        abs(int(predicted_parts[1]) - int(gold_parts[1]))
        if parsed and component_matches[0]
        else None
    )
    return {
        "has_reasoning_label": bool(REASONING_LABEL_RE.search(raw_text or "")),
        "has_edit_label": bool(labels),
        "bare_edit_only": bare_edit_only,
        "format_compliant": bare_edit_only or labeled_edit_only,
        "parse_success": parsed,
        "exact_match": parsed and predicted == gold,
        "edit_type_exact": component_matches[0],
        "position_exact": component_matches[1],
        "new_sequence_exact": component_matches[2],
        "edit_exact": all(component_matches),
        "position_error": position_error,
    }


def score_answer(
    predicted: Optional[str],
    row: Mapping[str, Any],
    raw_text: Optional[str],
    answer_format: str,
) -> Dict[str, Any]:
    if answer_format == "variant-record":
        return score_variant(predicted, row, raw_text)
    if answer_format == "local-edit":
        return score_local_edit(predicted, row, raw_text)
    raise ValueError(f"unsupported answer format: {answer_format}")


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lower = math.floor(point)
    upper = math.ceil(point)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - point) + ordered[upper] * (point - lower))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[Optional[float]]:
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_scored(
    scored: Sequence[
        Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    answer_format: str,
) -> Dict[str, Any]:
    def summarize_subset(items: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
        total = len(items)
        common_boolean_names = ("format_compliant", "parse_success", "exact_match")
        if answer_format == "variant-record":
            format_boolean_names = (
                "bare_variant_only",
                "assembly_exact",
                "chromosome_exact",
                "position_exact",
                "reference_allele_exact",
                "alternate_allele_exact",
                "locus_exact",
                "edit_exact",
            )
            error_name = "coordinate_error"
        else:
            format_boolean_names = (
                "bare_edit_only",
                "edit_type_exact",
                "position_exact",
                "new_sequence_exact",
                "edit_exact",
            )
            error_name = "position_error"
        boolean_names = common_boolean_names + format_boolean_names
        counts = {name: sum(bool(top[name]) for _, top, _ in items) for name in boolean_names}
        position_errors = [
            float(top[error_name])
            for _, top, _ in items
            if top[error_name] is not None
        ]
        pass_at_k = sum(any(bool(candidate["exact_match"]) for candidate in candidates) for _, _, candidates in items)
        result: Dict[str, Any] = {"examples": total}
        for name, count in counts.items():
            result[f"{name}_count"] = count
            result[f"{name}_rate"] = count / total if total else None
        result["exact_match_95ci_wilson"] = wilson_interval(counts["exact_match"], total)
        result["pass_at_k_exact_count"] = pass_at_k
        result["pass_at_k_exact_rate"] = pass_at_k / total if total else None
        if answer_format == "variant-record":
            mean_error_name = "mean_coordinate_error_when_chromosome_matches"
            median_error_name = "median_coordinate_error_when_chromosome_matches"
        else:
            mean_error_name = "mean_position_error_when_edit_type_matches"
            median_error_name = "median_position_error_when_edit_type_matches"
        result[mean_error_name] = (
            statistics.fmean(position_errors) if position_errors else None
        )
        result[median_error_name] = percentile(position_errors, 0.5)
        return result

    groups: Dict[str, Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]]]] = {
        "variant_type": defaultdict(list),
        "data_project": defaultdict(list),
        "effect_magnitude_max_abs_log2fc": defaultdict(list),
    }
    for row, top, candidates in scored:
        item = (row, top, candidates)
        groups["variant_type"][variant_type(row)].append(item)
        groups["data_project"][str(row.get("data_project") or "unknown")].append(item)
        groups["effect_magnitude_max_abs_log2fc"][effect_bin(row)].append(item)

    return {
        "answer_format": answer_format,
        "overall": summarize_subset(scored),
        "breakdowns": {
            group_name: {name: summarize_subset(items) for name, items in sorted(group.items())}
            for group_name, group in groups.items()
        },
    }


def resolve_chat_template(args: argparse.Namespace, profile: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    if args.chat_template is not None:
        if not args.chat_template.is_file():
            raise FileNotFoundError(f"Chat template does not exist: {args.chat_template}")
        template = args.chat_template.read_text(encoding="utf-8")
        if args.reasoning_effort is not None and "reasoning_effort" not in template:
            raise ValueError(
                "--reasoning-effort was set, but the explicit chat template does not consume "
                "reasoning_effort. Use the GPT-OSS model tokenizer template."
            )
        return template, str(args.chat_template.resolve())
    if args.chat_template_source == "model":
        return None, "model_tokenizer"
    parquet_file = pq.ParquetFile(args.input_parquet)
    template = parquet_chat_template(parquet_file)
    if template is None:
        raise ValueError(
            "--chat-template-source=parquet was requested, but the parquet has no chat_template metadata. "
            "Use --chat-template-source=model or --chat-template PATH."
        )
    if args.reasoning_effort is not None and "reasoning_effort" not in template:
        raise ValueError(
            "--reasoning-effort is not supported by the parquet chat template. "
            "For GPT-OSS, pass --chat-template-source=model."
        )
    return template, "parquet_schema_metadata"


def make_run_config(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    template: Optional[str],
    template_source: str,
) -> Dict[str, Any]:
    return {
        "format_version": 3,
        "answer_format": args.answer_format,
        "input": dict(profile),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "chat_template_source": template_source,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None,
        "cot_prompt": args.cot_prompt,
        "cot_prompt_version": COT_PROMPT_VERSION if args.cot_prompt else None,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "n": args.n,
            "seed": args.seed,
            "reasoning_effort": args.reasoning_effort,
        },
        "engine": {
            "dtype": args.dtype,
            "quantization": args.quantization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "cpu_offload_gb": args.cpu_offload_gb,
            "swap_space": args.swap_space,
            "trust_remote_code": args.trust_remote_code,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": args.enable_prefix_caching,
        },
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def comparable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    # Paths may be mounted through aliases; content hash and actual inference
    # settings are the compatibility boundary for a resumed prediction file.
    answer_format = config.get("answer_format")
    if answer_format == "canonical_grch37_variant_record":
        answer_format = "variant-record"
    return {
        "input_sha256": config.get("input", {}).get("sha256"),
        "selected_start_index": config.get("input", {}).get("selected_start_index"),
        "selected_rows": config.get("input", {}).get("selected_rows"),
        "answer_format": answer_format,
        "model": config.get("model"),
        "tokenizer": config.get("tokenizer"),
        "revision": config.get("revision"),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "chat_template_sha256": config.get("chat_template_sha256"),
        "cot_prompt": bool(config.get("cot_prompt", False)),
        "cot_prompt_version": config.get("cot_prompt_version"),
        "sampling": config.get("sampling"),
        "engine": config.get("engine"),
    }


def prepare_outputs(args: argparse.Namespace, config: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    config_path = args.output_dir / "run_config.json"
    metrics_path = args.output_dir / "metrics.json"

    if args.overwrite:
        for path in (predictions, config_path, metrics_path):
            if path.exists():
                path.unlink()
    elif predictions.exists() and not args.resume:
        raise FileExistsError(f"{predictions} exists; use --resume or --overwrite")

    if predictions.exists():
        if not config_path.exists():
            raise ValueError(f"Cannot safely resume {predictions}: run_config.json is missing")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if comparable_config(previous) != comparable_config(config):
            raise ValueError(
                "Existing predictions were created with an incompatible dataset/model/config. "
                "Choose another --output-dir or pass --overwrite."
            )
    else:
        atomic_write_json(config_path, config)
    return predictions, config_path, metrics_path


def read_prediction_records(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                row_index = int(record["row_index"])
            except Exception as exc:
                raise ValueError(f"Malformed prediction record at {path}:{line_number}: {exc}") from exc
            if row_index in records:
                raise ValueError(f"Duplicate row_index {row_index} in {path}")
            records[row_index] = record
    return records


def initialize_vllm(args: argparse.Namespace) -> Tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this Python environment. Install a CUDA-compatible vLLM build, "
            "then run this script from that environment."
        ) from exc

    llm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "enable_prefix_caching": args.enable_prefix_caching,
        "seed": args.seed,
    }
    # swap_space was removed in vLLM 0.26. Passing it through LLM's **kwargs
    # reaches EngineArgs and raises TypeError, so only forward it on versions
    # whose public LLM constructor still supports the option.
    if "swap_space" in inspect.signature(LLM.__init__).parameters:
        llm_kwargs["swap_space"] = args.swap_space
    optional = {
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "download_dir": args.download_dir,
    }
    llm_kwargs.update({key: value for key, value in optional.items() if value is not None})
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    return llm, sampling


def generate_pending(
    args: argparse.Namespace,
    answer_format: str,
    rows: Sequence[Mapping[str, Any]],
    existing: Mapping[int, Mapping[str, Any]],
    predictions_path: Path,
    template: Optional[str],
    llm: Any,
    sampling: Any,
    initial_scored: Sequence[
        Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    checkpoint_callback: Callable[
        [
            Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]],
            Mapping[str, Any],
        ],
        None,
    ],
) -> Dict[str, Any]:
    pending = [row for row in rows if int(row["row_index"]) not in existing]
    started = time.perf_counter()
    prompt_tokens = 0
    output_tokens = 0
    generated_examples = 0
    scored_progress = list(initial_scored)
    completed_at_last_save = len(existing)

    def current_runtime() -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {
            "resumed_examples": len(existing),
            "generated_examples_this_run": generated_examples,
            "generation_wall_seconds_this_run": elapsed,
            "prompt_tokens_this_run": prompt_tokens,
            "output_tokens_this_run": output_tokens,
            "examples_per_second_this_run": generated_examples / elapsed if elapsed else None,
            "tokens_per_second_this_run": (prompt_tokens + output_tokens) / elapsed if elapsed else None,
        }

    mode = "a" if predictions_path.exists() else "w"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        begin = 0
        while begin < len(pending):
            completed = len(existing) + generated_examples
            until_checkpoint = args.save_every - (completed % args.save_every)
            batch_size = min(
                args.request_chunk_size,
                until_checkpoint,
                len(pending) - begin,
            )
            batch = pending[begin : begin + batch_size]
            begin += batch_size
            conversations = [row["prompt_messages"] for row in batch]
            outputs = llm.chat(
                conversations,
                sampling_params=sampling,
                use_tqdm=not args.quiet_vllm,
                chat_template=template,
                add_generation_prompt=True,
                chat_template_kwargs=(
                    {"reasoning_effort": args.reasoning_effort}
                    if args.reasoning_effort is not None
                    else None
                ),
            )
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for a {len(batch)}-request batch")

            for row, request_output in zip(batch, outputs):
                candidates = []
                for rank, candidate in enumerate(request_output.outputs):
                    text = candidate.text
                    parsed_answer = extract_answer(text, answer_format)
                    candidate_record = {
                        "rank": rank,
                        "text": text,
                        "parsed_answer": parsed_answer,
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "output_tokens": len(candidate.token_ids or []),
                        "scores": score_answer(parsed_answer, row, text, answer_format),
                    }
                    # Keep the format-specific key convenient for downstream readers.
                    candidate_record[
                        "parsed_variant" if answer_format == "variant-record" else "parsed_edit"
                    ] = parsed_answer
                    candidates.append(candidate_record)
                    output_tokens += len(candidate.token_ids or [])
                prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or []
                prompt_tokens += len(prompt_token_ids)
                record = {
                    "row_index": int(row["row_index"]),
                    "pair_id": row["pair_id"],
                    "prompt_tokens": len(prompt_token_ids),
                    "candidates": candidates,
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                generated_examples += 1
                candidate_scores = [candidate["scores"] for candidate in candidates]
                scored_progress.append((row, candidate_scores[0], candidate_scores))

                completed = len(existing) + generated_examples
                if completed % args.save_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    checkpoint_callback(scored_progress, current_runtime())
                    completed_at_last_save = completed

        completed = len(existing) + generated_examples
        if completed != completed_at_last_save:
            handle.flush()
            os.fsync(handle.fileno())
            checkpoint_callback(scored_progress, current_runtime())

    return current_runtime()


def score_prediction_records(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    answer_format: str,
) -> Tuple[List[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]], Dict[str, Any]]:
    scored = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    finish_reasons: Counter[str] = Counter()
    top1_length_truncations = 0
    for row in rows:
        idx = int(row["row_index"])
        if idx not in records:
            raise ValueError(f"Missing prediction for row_index {idx}")
        record = records[idx]
        if str(record.get("pair_id")) != str(row["pair_id"]):
            raise ValueError(f"pair_id mismatch for row_index {idx}")
        raw_candidates = record.get("candidates") or []
        if not raw_candidates:
            raise ValueError(f"No candidates recorded for row_index {idx}")
        candidate_scores = []
        for rank, candidate in enumerate(raw_candidates):
            text = str(candidate.get("text") or "")
            parsed_answer = extract_answer(text, answer_format)
            candidate_scores.append(score_answer(parsed_answer, row, text, answer_format))
            total_output_tokens += int(candidate.get("output_tokens") or 0)
            finish_reason = str(candidate.get("finish_reason") or "unknown")
            finish_reasons[finish_reason] += 1
            if rank == 0 and finish_reason == "length":
                top1_length_truncations += 1
        total_prompt_tokens += int(record.get("prompt_tokens") or 0)
        scored.append((row, candidate_scores[0], candidate_scores))
    return scored, {
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "finish_reason_counts_all_candidates": dict(finish_reasons),
        "top1_length_truncation_count": top1_length_truncations,
        "top1_length_truncation_rate": top1_length_truncations / len(rows) if rows else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows, profile = load_rows(args.input_parquet, args.start_index, args.max_examples)
    args.answer_format = resolve_answer_format(args.answer_format, profile)
    if args.cot_prompt:
        configure_cot_prompts(rows)
    validation = validate_rows(rows, args.answer_format)
    profile["validation"] = validation

    if args.validate_only:
        print(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True))
        return

    template, template_source = resolve_chat_template(args, profile)
    config = make_run_config(args, profile, template, template_source)
    predictions_path, _, metrics_path = prepare_outputs(args, config)
    existing = read_prediction_records(predictions_path)
    selected_indices = {int(row["row_index"]) for row in rows}
    unexpected = sorted(set(existing) - selected_indices)
    if unexpected:
        raise ValueError(f"Prediction file contains row indices outside this selection, e.g. {unexpected[:5]}")

    pending_count = len(rows) - len(existing)
    if pending_count:
        existing_rows = [row for row in rows if int(row["row_index"]) in existing]
        initial_scored, _ = score_prediction_records(
            existing_rows, existing, args.answer_format
        )

        def report_checkpoint(
            current_scored: Sequence[
                Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
            ],
            current_runtime: Mapping[str, Any],
        ) -> None:
            checkpoint_metrics = summarize_scored(current_scored, args.answer_format)
            completed = len(current_scored)
            checkpoint_metrics.update(
                {
                    "format_version": 3,
                    "status": "complete" if completed == len(rows) else "in_progress",
                    "examples_completed": completed,
                    "examples_total": len(rows),
                    "dataset": profile,
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "candidate_count": args.n,
                    "runtime": dict(current_runtime),
                    "artifacts": {
                        "predictions": str(predictions_path.resolve()),
                        "run_config": str((args.output_dir / "run_config.json").resolve()),
                    },
                }
            )
            atomic_write_json(metrics_path, checkpoint_metrics)
            overall = checkpoint_metrics["overall"]
            if args.answer_format == "variant-record":
                component_text = (
                    f"locus={overall['locus_exact_rate']:.4%} | "
                    f"alleles={overall['edit_exact_rate']:.4%} | "
                )
            else:
                component_text = (
                    f"type={overall['edit_type_exact_rate']:.4%} | "
                    f"position={overall['position_exact_rate']:.4%} | "
                )
            print(
                "checkpoint metrics: "
                f"{completed}/{len(rows)} | "
                f"exact={overall['exact_match_rate']:.4%} | "
                f"parsed={overall['parse_success_rate']:.4%} | "
                f"{component_text}"
                f"pass@{args.n}={overall['pass_at_k_exact_rate']:.4%}",
                file=sys.stderr,
                flush=True,
            )

        print(
            f"Loading {args.model!r}; {pending_count} of {len(rows)} examples require generation.",
            file=sys.stderr,
            flush=True,
        )
        llm, sampling = initialize_vllm(args)
        runtime = generate_pending(
            args,
            args.answer_format,
            rows,
            existing,
            predictions_path,
            template,
            llm,
            sampling,
            initial_scored,
            report_checkpoint,
        )
    else:
        runtime = {
            "resumed_examples": len(existing),
            "generated_examples_this_run": 0,
            "generation_wall_seconds_this_run": 0.0,
            "prompt_tokens_this_run": 0,
            "output_tokens_this_run": 0,
            "examples_per_second_this_run": None,
            "tokens_per_second_this_run": None,
        }

    records = read_prediction_records(predictions_path)
    scored, token_totals = score_prediction_records(rows, records, args.answer_format)
    metrics = summarize_scored(scored, args.answer_format)
    metrics.update(
        {
            "format_version": 3,
            "status": "complete",
            "examples_completed": len(rows),
            "examples_total": len(rows),
            "dataset": profile,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "candidate_count": args.n,
            "runtime": runtime,
            "recorded_token_totals": token_totals,
            "artifacts": {
                "predictions": str(predictions_path.resolve()),
                "run_config": str((args.output_dir / "run_config.json").resolve()),
            },
        }
    )
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
