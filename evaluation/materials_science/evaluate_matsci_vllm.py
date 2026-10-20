#!/usr/bin/env python3
"""Run resumable vLLM evaluation on the Materials Project temporal split.

Scores the candidate reduced formula, not the free-form reasoning.  For the
``test_detailed.parquet`` variant it also scores the six requested unit-cell
parameters.  vLLM is imported lazily, so validation and metric recomputation
work on CPU-only nodes.  Sanitized, composition-blind CIF context can be added
with ``--include-cif``. By default it uses the task-first prompt format used
for SFT; the legacy prepend-to-user-message layout remains available for
reproducing older runs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
# The task-first CIF prompt builder lives with the distillation code so that
# evaluation prompts cannot drift from the SFT data format.
DISTILLATION_DIR = SCRIPT_DIR.parents[1] / "distillation" / "materials_science"
for helper_dir in (SCRIPT_DIR, DISTILLATION_DIR):
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

from formulas import canonical_formula


CELL_FIELDS = (
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
)
LENGTH_FIELDS = CELL_FIELDS[:3]
ANGLE_FIELDS = CELL_FIELDS[3:]
REQUIRED_COLUMNS = ("id", "split", "messages", "metadata_json", "reference_answer_json")
OPTIONAL_COLUMNS = ("task", "source") + CELL_FIELDS
REASONING_RE = re.compile(r"(?im)^\s*Reasoning\s*:")
CANDIDATE_RE = re.compile(
    r"(?i)(?<![A-Za-z])(?:thus[^\S\r\n]+)?(?:my[^\S\r\n]+)?\*{0,3}"
    r"Candi(?:date|diate|ate)\*{0,3}(?:[^\S\r\n]+(?:material|formula))?"
    r"(?:[^\S\r\n]+is)?[^\S\r\n]*:\*{0,3}[^\S\r\n]*([^\r\n]*)"
)
# With ``skip_special_tokens=True``, GPT-OSS may expose its final-channel name
# immediately adjacent to the message content (for example ``finalNaCl``).
# Keep this anchored to the entire response so an ordinary occurrence of the
# word "final" in free-form reasoning cannot be mistaken for an answer.
GPT_OSS_FINAL_RE = re.compile(
    r"(?s)^\s*(?:<\|channel\|>\s*)?final"
    r"(?:\s*<\|message\|>\s*|\s*:\s*|\s+)?(.+?)\s*$"
)
MARKDOWN_LABEL_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?\*{1,2}(Reasoning|Candidate)\s*:\*{1,2}\s*"
)
MARKDOWN_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?\*{1,2}(Reasoning|Candidate)\*{1,2}\s*$"
)
MARKDOWN_INLINE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[-+>]\s*)?\*{1,3}(Reasoning|Candidate)\*{1,3}\s+(?!:)"
)
FORMULA_TAG_RE = re.compile(r"\\?</?formula\s*>", re.IGNORECASE)
CANDIDATE_TAG_RE = re.compile(r"<candidate\s*>(.*?)</candidate\s*>", re.IGNORECASE)
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_SUBSCRIPT_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE)
LATEX_SUBSCRIPT_RE = re.compile(
    r"\$?_\s*(?:\{([^{}]+)\}|([0-9]+(?:\.[0-9]+)?))\$?"
)
LATEX_WRAPPER_RE = re.compile(
    r"\\(?:boldsymbol|boxed|ce|mathrm|mathbf|mathit|text|textrm)\s*\{([^{}]*)\}"
)
LATEX_RM_RE = re.compile(r"\{\s*\\rm\s+([^{}]*)\}")
LATEX_DECLARATION_RE = re.compile(r"\\(?:displaystyle|textstyle)\b")
CANDIDATE_TEMPLATE = (
    "Use this response template exactly:\n"
    "Reasoning: <reasoning trace>\n"
    "Candidate: <formula>"
)
CANDIDATE_ONLY_INSTRUCTION = (
    "Return only the candidate formula, with no reasoning or additional text.\n"
    "Candidate: "
)
UNICODE_SUBSCRIPT_TRANSLATION =str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CELL_RES = {
    field: re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*({FLOAT_RE})")
    for field in CELL_FIELDS
}


def parse_args(
    argv: Optional[Sequence[str]] = None,
    configure_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    description: Optional[str] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description or "Evaluate a local model with vLLM on the materials temporal-split parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    data = parser.add_argument_group("data selection")
    data.add_argument("--start-index", type=int, default=0)
    data.add_argument("--max-examples", type=int, default=None)
    prompt = parser.add_argument_group("prompt fidelity")
    prompt.add_argument("--chat-template-source", choices=("parquet", "model"), default="model")
    prompt.add_argument("--chat-template", type=Path)
    prompt.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    prompt.add_argument(
        "--include-cif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the same sanitized, composition-blind reference CIF context "
            "used by evaluate_matsci_api_cif.py."
        ),
    )
    prompt.add_argument(
        "--cif-prompt-layout",
        choices=("sft_task_first", "legacy"),
        default="sft_task_first",
        help=(
            "CIF prompt layout when --include-cif is set: sft_task_first matches "
            "the CIF SFT data; legacy prepends CIF context to the original user message."
        ),
    )
    prompt.add_argument(
        "--candidate-only-prompt",
        action="store_true",
        help=(
            "Replace the standard reasoning/Candidate response template with the "
            "formula-only prompt used by candidate-only SFT. Applied before any "
            "CIF context is attached."
        ),
    )
    prompt.add_argument(
        "--max-cif-chars",
        type=int,
        help="Optional character limit for sanitized CIF context (requires --include-cif).",
    )
    prompt.add_argument(
        "--selected-input-prompt",
        action="store_true",
        help=(
            "Use the API selected-input prompt pipeline, including property-line "
            "ablations and optional distilled reasoning."
        ),
    )
    for name, label in (
        ("material-family", "Material family"),
        ("stability", "Stability"),
        ("electronic-character", "Electronic character"),
        ("formation-energy", "Formation energy"),
    ):
        prompt.add_argument(
            f"--include-{name}",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"Include the {label!r} constraint in selected-input prompts.",
        )
    prompt.add_argument("--distilled-reasoning-jsonl", type=Path)
    prompt.add_argument(
        "--distilled-reasoning-level",
        choices=("none", "quarter", "half", "full"),
        default="none",
    )
    generation = parser.add_argument_group("generation")
    generation.add_argument("--request-chunk-size", type=int, default=1024)
    generation.add_argument("--max-tokens", type=int, default=512)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--top-k", type=int, default=-1)
    generation.add_argument("--min-p", type=float, default=0.0)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--presence-penalty", type=float, default=0.0)
    generation.add_argument("--frequency-penalty", type=float, default=0.0)
    generation.add_argument("--n", type=int, default=1)
    generation.add_argument("--seed", type=int, default=13)
    engine = parser.add_argument_group("vLLM engine")
    engine.add_argument("--tokenizer")
    engine.add_argument("--revision")
    engine.add_argument("--tokenizer-revision")
    engine.add_argument("--dtype", default="auto", choices=("auto", "half", "float16", "bfloat16", "float", "float32"))
    engine.add_argument("--quantization")
    engine.add_argument("--tensor-parallel-size", type=int, default=1)
    engine.add_argument("--pipeline-parallel-size", type=int, default=1)
    engine.add_argument("--gpu-memory-utilization", type=float, default=.90)
    engine.add_argument("--max-model-len", type=int)
    engine.add_argument("--max-num-seqs", type=int)
    engine.add_argument("--cpu-offload-gb", type=float, default=0.0)
    engine.add_argument("--swap-space", type=float, default=4.0)
    engine.add_argument("--download-dir")
    engine.add_argument("--trust-remote-code", action="store_true")
    engine.add_argument("--enforce-eager", action="store_true")
    engine.add_argument("--disable-custom-all-reduce", action="store_true")
    engine.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    output = parser.add_argument_group("output/recovery")
    output.add_argument("--save-every", type=int, default=1000)
    output.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument(
        "--refresh-stored-scores",
        action="store_true",
        help=(
            "Recompute parsed_formula, parsed_geometry, and scores in existing "
            "prediction records using the current parser. Raw generated text is unchanged."
        ),
    )
    output.add_argument("--quiet-vllm", action="store_true")
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    if args.start_index < 0 or (args.max_examples is not None and args.max_examples <= 0):
        parser.error("indices/counts must be positive (start-index may be zero)")
    if min(args.request_chunk_size, args.max_tokens, args.save_every, args.n) <= 0:
        parser.error("chunk size, max tokens, save interval, and n must be positive")
    if args.n > 1 and args.temperature <= 0:
        parser.error("--n > 1 requires --temperature > 0")
    if args.max_cif_chars is not None and args.max_cif_chars <= 0:
        parser.error("--max-cif-chars must be positive")
    if args.max_cif_chars is not None and not args.include_cif:
        parser.error("--max-cif-chars requires --include-cif")
    reasoning_enabled = args.distilled_reasoning_level != "none"
    if reasoning_enabled != bool(args.distilled_reasoning_jsonl):
        parser.error(
            "--distilled-reasoning-jsonl must be specified exactly when "
            "--distilled-reasoning-level is quarter, half, or full"
        )
    if reasoning_enabled and not args.selected_input_prompt:
        parser.error("distilled reasoning requires --selected-input-prompt")
    if reasoning_enabled and not all((
        args.include_material_family,
        args.include_stability,
        args.include_electronic_character,
        args.include_formation_energy,
        args.include_cif,
    )):
        parser.error("distilled reasoning requires all five input properties")
    if not args.validate_only and not args.model:
        parser.error("--model is required unless --validate-only is used")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if args.overwrite:
        args.resume = False
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_chat_template(parquet_file: pq.ParquetFile) -> Optional[str]:
    value = (parquet_file.schema_arrow.metadata or {}).get(b"chat_template")
    return value.decode("utf-8") if value else None


def _json_object(value: Any, label: str, row_index: int) -> Dict[str, Any]:
    try:
        result = json.loads(value) if isinstance(value, str) else dict(value)
    except Exception as exc:
        raise ValueError(f"row {row_index}: invalid {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"row {row_index}: {label} is not an object")
    return result


def load_rows(path: Path, start_index: int, max_examples: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {path}")
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(REQUIRED_COLUMNS) - available)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {', '.join(missing)}")
    total = parquet_file.metadata.num_rows
    if start_index >= total:
        raise ValueError(f"--start-index {start_index} is outside a {total}-row parquet")
    end = min(total, start_index + max_examples) if max_examples else total
    columns = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in available]
    rows = parquet_file.read(columns=columns).slice(start_index, end - start_index).to_pylist()
    for offset, row in enumerate(rows):
        idx = start_index + offset
        row["row_index"] = idx
        row["metadata"] = _json_object(row["metadata_json"], "metadata_json", idx)
        row["reference_answer"] = _json_object(row["reference_answer_json"], "reference_answer_json", idx)
        messages = row["messages"] or []
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {idx}: messages must be a non-empty list")
        # The final assistant message is the gold answer and must never be sent to the model.
        row["prompt_messages"] = messages[:-1] if messages[-1].get("role") == "assistant" else messages
        row["gold_formula"] = str(row["reference_answer"].get("candidate_formula") or row["metadata"].get("formula") or "")
        row["prompt_variant"] = str(row["metadata"].get("prompt_variant") or "basic")
        row["detailed"] = row["prompt_variant"] == "detailed_geometry"
    metadata = parquet_file.schema_arrow.metadata or {}
    profile = {
        "path": str(path.resolve()), "sha256": file_sha256(path), "parquet_rows": total,
        "selected_start_index": start_index, "selected_rows": len(rows),
        "row_groups": parquet_file.metadata.num_row_groups,
        "schema_columns": len(parquet_file.schema_arrow.names),
        "embedded_chat_template": b"chat_template" in metadata,
    }
    return rows, profile


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids: set[str] = set()
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        idx, row_id = row["row_index"], str(row["id"])
        if row_id in ids:
            raise ValueError(f"duplicate id {row_id!r} in selected data")
        ids.add(row_id)
        if not row["prompt_messages"] or any(m.get("role") == "assistant" for m in row["prompt_messages"]):
            raise ValueError(f"row {idx}: prompt contains no request or leaks an assistant answer")
        try:
            _, _, formula_system = canonical_formula(row["gold_formula"])
        except Exception as exc:
            raise ValueError(f"row {idx}: invalid gold formula {row['gold_formula']!r}: {exc}") from exc
        expected_system = row["metadata"].get("chemical_system")
        if expected_system and set(formula_system.split("-")) != set(str(expected_system).split("-")):
            raise ValueError(f"row {idx}: gold formula and chemical system disagree")
        if row["detailed"] and any(row.get(field) is None for field in CELL_FIELDS):
            raise ValueError(f"row {idx}: detailed example lacks unit-cell values")
        counts["split"][str(row["split"])] += 1
        counts["prompt_variant"][str(row["prompt_variant"])] += 1
        counts["material_family"][str(row["metadata"].get("material_family") or "unknown")] += 1
        counts["chemical_system_arity"][str(len(formula_system.split("-")))] += 1
    return {"rows": len(rows), **{key: dict(value) for key, value in counts.items()}}


def configure_cif_prompts(
    rows: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    max_cif_chars: Optional[int] = None,
    layout: str = "sft_task_first",
) -> None:
    """Attach sanitized CIF context in the requested generation-prompt layout.

    ``sft_task_first`` imports the training prompt builder so evaluation cannot
    drift from the prompt format used to construct the fine-tuning data.
    ``legacy`` reproduces the original CIF transform, which prepends the CIF
    context to the first user message.
    """
    # When this file is executed as a script its module name is ``__main__``;
    # expose that instance under the import name expected by the API module so
    # importing the shared CIF helpers does not execute this file a second time.
    sys.modules.setdefault("evaluate_matsci_vllm", sys.modules[__name__])
    from evaluate_matsci_api_cif import cif_prompt_messages, configure_cif_context

    configure_cif_context(rows, profile, max_cif_chars=max_cif_chars, layout=layout)
    if layout == "legacy":
        for row in rows:
            row["_generation_prompt_messages"] = cif_prompt_messages(row)
        return
    if layout == "sft_task_first":
        from generate_distillation_data_cif import make_sft_prompt_messages

        for row in rows:
            row["_generation_prompt_messages"] = make_sft_prompt_messages(
                row, include_cif=True
            )
        return
    raise ValueError(f"unknown CIF prompt layout: {layout}")


def candidate_only_prompt(messages: Any, example_id: Any) -> List[Dict[str, Any]]:
    """Replace the reasoning/Candidate template with the stage-1 formula-only one."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Example {example_id} has no messages to transform.")

    transformed = [dict(message) for message in messages]
    user_messages = [message for message in transformed if message.get("role") == "user"]
    if not user_messages:
        raise ValueError(f"Example {example_id} has no user message.")

    replaced = False
    for message in user_messages:
        content = str(message.get("content") or "")
        if CANDIDATE_TEMPLATE in content:
            message["content"] = content.replace(CANDIDATE_TEMPLATE, CANDIDATE_ONLY_INSTRUCTION)
            replaced = True
    if not replaced:
        raise ValueError(f"Example {example_id} does not contain the expected response template.")
    return transformed


def configure_candidate_only_prompts(rows: Sequence[Dict[str, Any]]) -> None:
    """Apply the candidate-only (stage-1) SFT prompt transform to generation messages."""
    for row in rows:
        row["prompt_messages"] = candidate_only_prompt(
            row["prompt_messages"], row.get("id")
        )


def configure_selected_input_prompts(
    rows: Sequence[Dict[str, Any]], profile: Dict[str, Any], args: argparse.Namespace
) -> None:
    """Apply the exact prompt transforms used by the API property ablation."""
    sys.modules.setdefault("evaluate_matsci_vllm", sys.modules[__name__])
    from evaluate_matsci_api_cif import (
        augment_with_distilled_reasoning,
        configure_cif_context,
        configure_distilled_reasoning,
        configure_input_properties,
        filter_property_lines,
    )
    from generate_distillation_data_cif import make_sft_prompt_messages

    configure_input_properties(profile, args)
    if args.include_cif:
        configure_cif_context(rows, profile, max_cif_chars=args.max_cif_chars)
    if args.distilled_reasoning_level != "none":
        configure_distilled_reasoning(
            rows,
            profile,
            args.distilled_reasoning_jsonl,
            args.distilled_reasoning_level,
        )
    for row in rows:
        messages = make_sft_prompt_messages(row, include_cif=args.include_cif)
        messages = filter_property_lines(messages, args)
        row["_generation_prompt_messages"] = augment_with_distilled_reasoning(
            messages, row
        )


def normalize_response_markup(text: str) -> str:
    """Normalize presentation-only markup before extracting a formula.

    Model responses commonly render stoichiometry with Unicode, HTML, or
    LaTeX subscripts, and sometimes bold the requested field labels.  These
    forms carry the same formula content as plain ASCII and should therefore
    receive the same score.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(r"\/", "/").replace(r"\nCandidate:", "\nCandidate:")
    text = text.translate(UNICODE_SUBSCRIPT_TRANSLATION).replace("․", ".")
    text = HTML_BREAK_RE.sub("\n", text)
    text = CANDIDATE_TAG_RE.sub(r"Candidate: \1", text)
    text = HTML_SUBSCRIPT_RE.sub(r"\1", text)
    text = FORMULA_TAG_RE.sub("", text)
    text = text.replace(r"\_", "_")
    text = LATEX_SUBSCRIPT_RE.sub(lambda match: match.group(1) or match.group(2), text)
    # Resolve innermost wrappers repeatedly so constructs such as
    # ``\boxed{\mathrm{NaCl}}`` become plain ``NaCl``.
    while True:
        unwrapped = LATEX_WRAPPER_RE.sub(r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped
    text = LATEX_RM_RE.sub(r"\1", text)
    text = LATEX_DECLARATION_RE.sub("", text)
    text = (
        text.replace(r"\(", "")
        .replace(r"\)", "")
        .replace(r"\[", "")
        .replace(r"\]", "")
        .replace(r"\,", "")
        .replace(r"\!", "")
    )
    text = MARKDOWN_HEADING_RE.sub(r"\1:", text)
    text = MARKDOWN_INLINE_HEADING_RE.sub(r"\1: ", text)
    return MARKDOWN_LABEL_RE.sub(r"\1: ", text)


def _formula_from_candidate(value: str, allow_trailing_text: bool = True) -> Optional[str]:
    value = normalize_response_markup(value).strip()
    # Markdown emphasis/code and TeX math delimiters are presentation, not
    # composition syntax. Remove them before isolating the candidate token.
    value = (
        value.replace("`", "")
        .replace("*", "")
        .replace("_", "")
        .strip("#>•- \\ \t")
    )
    token = (
        re.split(r"\s+|[—–]|\((?:=|formula\b)", value, maxsplit=1)[0]
        if value and allow_trailing_text
        else value
    )
    token = token.strip().strip("${}<>").rstrip(".,;:!?")
    token = token.replace("[", "(").replace("]", ")")
    token = re.sub(r"(?<=[A-Za-z])\((\d+(?:\.\d+)?)\)", r"\1", token)
    if not token:
        return None
    try:
        canonical_formula(token)
        return token
    except Exception:
        if allow_trailing_text:
            # Occasionally a misspelled first formula is immediately corrected
            # in parentheses, e.g. ``BhDiErZn (BiDyErZn)``.
            correction = re.search(r"\(([A-Za-z][A-Za-z0-9./]*)\)", value)
            if correction:
                try:
                    canonical_formula(correction.group(1))
                    return correction.group(1)
                except Exception:
                    pass
        return None


def parse_response(text: str) -> Dict[str, Any]:
    text = normalize_response_markup(text)
    matches = list(CANDIDATE_RE.finditer(text))
    formula = None
    # Later labels supersede earlier drafts and, for serialized GPT-OSS
    # transcripts, avoid treating the analysis template's ``<formula>`` as
    # the answer. Keep looking if a later label itself is malformed.
    for match in reversed(matches):
        values = [match.group(1)]
        if not match.group(1).strip():
            # Accommodate a heading followed by a fenced or unfenced formula.
            values.extend(
                line.strip()
                for line in text[match.end():].splitlines()
                if line.strip()
            )
        for value in values[:4]:
            formula = _formula_from_candidate(value)
            if formula is not None:
                break
        if formula is not None:
            break
    if formula is None:
        final_match = GPT_OSS_FINAL_RE.fullmatch(text)
        if final_match:
            formula = _formula_from_candidate(
                final_match.group(1), allow_trailing_text=False
            )
    # A permissive fallback preserves scoring when the formula is present but the
    # requested label is missing; template compliance is measured separately.
    if formula is None:
        for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
            if (formula := _formula_from_candidate(line, allow_trailing_text=False)) is not None:
                break
            emphasized = re.findall(r"\*{1,3}([^*\r\n]+)\*{1,3}", line)
            for value in reversed(emphasized):
                if (formula := _formula_from_candidate(value, allow_trailing_text=False)) is not None:
                    break
            if formula is not None:
                break
    geometry: Dict[str, Optional[float]] = {}
    for field, pattern in CELL_RES.items():
        field_match = pattern.search(text)
        geometry[field] = float(field_match.group(1)) if field_match else None
    return {
        "formula": formula,
        "geometry": geometry,
        "has_reasoning_label": bool(REASONING_RE.search(text)),
        "has_candidate_label": bool(matches),
    }


def _permutation_invariant_cell_errors(pred: Mapping[str, float], gold: Mapping[str, float]) -> Dict[str, float]:
    """Choose the axis permutation minimizing normalized length/angle error.

    Under axis permutation, alpha is opposite a, beta opposite b, and gamma
    opposite c, so lengths and their opposite angles are permuted together.
    """
    pred_pairs = [(pred[LENGTH_FIELDS[i]], pred[ANGLE_FIELDS[i]]) for i in range(3)]
    gold_pairs = [(gold[LENGTH_FIELDS[i]], gold[ANGLE_FIELDS[i]]) for i in range(3)]
    best: Optional[Tuple[float, List[float], List[float], List[float]]] = None
    for permutation in itertools.permutations(range(3)):
        length_abs = [abs(pred_pairs[permutation[i]][0] - gold_pairs[i][0]) for i in range(3)]
        length_rel = [length_abs[i] / gold_pairs[i][0] for i in range(3)]
        angle_abs = [abs(pred_pairs[permutation[i]][1] - gold_pairs[i][1]) for i in range(3)]
        objective = sum(length_rel) + sum(value / 180.0 for value in angle_abs)
        candidate = (objective, length_abs, length_rel, angle_abs)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "cell_length_mae_angstrom": statistics.fmean(best[1]),
        "cell_length_mape": statistics.fmean(best[2]),
        "cell_angle_mae_degrees": statistics.fmean(best[3]),
        "cell_max_length_relative_error": max(best[2]),
        "cell_max_angle_absolute_error_degrees": max(best[3]),
    }


def score_response(text: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = parse_response(text)
    predicted_key = predicted_display = predicted_system = None
    if parsed["formula"] is not None:
        try:
            predicted_key, predicted_display, predicted_system = canonical_formula(parsed["formula"])
        except Exception:
            pass
    gold_key, gold_display, gold_system = canonical_formula(row["gold_formula"])
    detailed = bool(row["detailed"])
    geometry_complete = all(parsed["geometry"][field] is not None for field in CELL_FIELDS)
    score: Dict[str, Any] = {
        "has_reasoning_label": parsed["has_reasoning_label"],
        "has_candidate_label": parsed["has_candidate_label"],
        "format_compliant": parsed["has_reasoning_label"] and parsed["has_candidate_label"] and (not detailed or geometry_complete),
        "parse_success": predicted_key is not None,
        "predicted_formula": predicted_display,
        "gold_formula": gold_display,
        "exact_formula": predicted_key == gold_key if predicted_key else False,
        "chemical_system_match": predicted_system == gold_system if predicted_system else False,
        "geometry_required": detailed,
        "geometry_complete": geometry_complete if detailed else None,
    }
    if detailed and geometry_complete:
        gold_geometry = {field: float(row[field]) for field in CELL_FIELDS}
        errors = _permutation_invariant_cell_errors(parsed["geometry"], gold_geometry)
        score.update(errors)
        score["cell_within_1pct_1deg"] = errors["cell_max_length_relative_error"] <= .01 and errors["cell_max_angle_absolute_error_degrees"] <= 1.0
        score["cell_within_5pct_5deg"] = errors["cell_max_length_relative_error"] <= .05 and errors["cell_max_angle_absolute_error_degrees"] <= 5.0
    elif detailed:
        score.update({
            "cell_length_mae_angstrom": None, "cell_length_mape": None,
            "cell_angle_mae_degrees": None, "cell_max_length_relative_error": None,
            "cell_max_angle_absolute_error_degrees": None,
            "cell_within_1pct_1deg": False, "cell_within_5pct_5deg": False,
        })
    score["exact_formula_and_cell_within_1pct_1deg"] = score["exact_formula"] and score.get("cell_within_1pct_1deg", False) if detailed else None
    score["exact_formula_and_cell_within_5pct_5deg"] = score["exact_formula"] and score.get("cell_within_5pct_5deg", False) if detailed else None
    score["parsed_geometry"] = parsed["geometry"] if detailed else None
    return score


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[Optional[float]]:
    if not total:
        return [None, None]
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / (1 + z2 / total)
    return [center - radius, center + radius]


def _aggregate(scored: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"examples": len(scored)}
    booleans = (
        "has_reasoning_label", "has_candidate_label", "format_compliant", "parse_success",
        "exact_formula", "chemical_system_match", "geometry_complete", "cell_within_1pct_1deg",
        "cell_within_5pct_5deg", "exact_formula_and_cell_within_1pct_1deg",
        "exact_formula_and_cell_within_5pct_5deg",
    )
    for name in booleans:
        values = [bool(top[name]) for _, top, _ in scored if top.get(name) is not None]
        if values:
            success = sum(values)
            result[name] = {"count": success, "eligible": len(values), "rate": success / len(values), "wilson_95": wilson_interval(success, len(values))}
    for name in ("cell_length_mae_angstrom", "cell_length_mape", "cell_angle_mae_degrees"):
        values = [float(top[name]) for _, top, _ in scored if top.get(name) is not None]
        if values:
            result[name] = {"mean": statistics.fmean(values), "median": statistics.median(values), "eligible": len(values)}
    max_k = max((len(all_scores) for _, _, all_scores in scored), default=0)
    for metric in ("exact_formula", "exact_formula_and_cell_within_1pct_1deg", "exact_formula_and_cell_within_5pct_5deg"):
        eligible = [(all_scores) for _, top, all_scores in scored if top.get(metric) is not None]
        if eligible:
            result[f"pass_at_k/{metric}"] = {
                str(k): sum(any(candidate.get(metric) for candidate in candidates[:k]) for candidates in eligible) / len(eligible)
                for k in range(1, max_k + 1)
            }
    return result


def summarize_scored(scored: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, list]] = {
        "prompt_variant": defaultdict(list), "material_family": defaultdict(list),
        "chemical_system_arity": defaultdict(list),
    }
    for item in scored:
        row = item[0]
        _, _, system = canonical_formula(row["gold_formula"])
        groups["prompt_variant"][str(row["prompt_variant"])].append(item)
        groups["material_family"][str(row["metadata"].get("material_family") or "unknown")].append(item)
        groups["chemical_system_arity"][str(len(system.split("-")))].append(item)
    return {
        "top1_and_pass_at_k": _aggregate(scored),
        "breakdowns": {name: {key: _aggregate(value) for key, value in sorted(group.items())} for name, group in groups.items()},
    }


def resolve_chat_template(args: argparse.Namespace, profile: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    if args.chat_template:
        if not args.chat_template.is_file():
            raise FileNotFoundError(f"Chat template does not exist: {args.chat_template}")
        return args.chat_template.read_text(encoding="utf-8"), str(args.chat_template.resolve())
    if args.chat_template_source == "model":
        return None, "model_tokenizer"
    template = parquet_chat_template(pq.ParquetFile(args.input_parquet))
    if template is None:
        raise ValueError("parquet has no chat_template metadata; use --chat-template-source=model")
    return template, "parquet_schema_metadata"


def make_run_config(args: argparse.Namespace, profile: Mapping[str, Any], template: Optional[str], source: str) -> Dict[str, Any]:
    config = {
        "format_version": 1, "input": dict(profile), "model": args.model, "tokenizer": args.tokenizer,
        "revision": args.revision, "tokenizer_revision": args.tokenizer_revision,
        "chat_template_source": source,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None,
        "sampling": {key: getattr(args, key) for key in ("max_tokens", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "n", "seed", "reasoning_effort")},
        "engine": {key: getattr(args, key) for key in ("dtype", "quantization", "tensor_parallel_size", "pipeline_parallel_size", "gpu_memory_utilization", "max_model_len", "max_num_seqs", "cpu_offload_gb", "swap_space", "trust_remote_code", "enforce_eager", "enable_prefix_caching")},
    }
    if (getattr(args, "include_cif", False)
            or getattr(args, "candidate_only_prompt", False)
            or getattr(args, "selected_input_prompt", False)):
        config["prompt_context"] = {
            "include_cif": bool(getattr(args, "include_cif", False)),
            "layout": args.cif_prompt_layout if args.include_cif else None,
            "max_cif_characters": getattr(args, "max_cif_chars", None),
            "candidate_only_prompt": bool(getattr(args, "candidate_only_prompt", False)),
        }
        if getattr(args, "selected_input_prompt", False):
            config["prompt_context"].update({
                "selected_input_prompt": True,
                "input_properties": {
                key: bool(getattr(args, f"include_{key}"))
                for key in (
                    "material_family", "stability", "electronic_character",
                    "formation_energy", "cif",
                )
                },
                "distilled_reasoning_level": args.distilled_reasoning_level,
            })
    return config


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def comparable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    def normalize_local_resource(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        path = Path(value).expanduser()
        if path.is_absolute() and path.exists():
            return str(path.resolve())
        return value

    prompt_context = config.get("prompt_context")
    if isinstance(prompt_context, Mapping):
        prompt_context = dict(prompt_context)
        # Older CoT runs omitted this false-valued field. Treat omission and
        # explicit false as the same prompt contract during resume checks.
        prompt_context.setdefault("candidate_only_prompt", False)
    return {
        "input_sha256": config.get("input", {}).get("sha256"),
        "input_source_sha256": config.get("input", {}).get(
            "source_sha256", config.get("input", {}).get("sha256")
        ),
        "selected_start_index": config.get("input", {}).get("selected_start_index"),
        "selected_rows": config.get("input", {}).get("selected_rows"),
        "model": normalize_local_resource(config.get("model")),
        "tokenizer": normalize_local_resource(config.get("tokenizer")),
        **{key: config.get(key) for key in ("revision", "tokenizer_revision", "chat_template_sha256", "sampling", "engine")},
        "prompt_context": prompt_context,
    }


def resume_config_compatible(
    existing_config: Mapping[str, Any],
    requested_config: Mapping[str, Any],
) -> bool:
    """Allow an existing run to grow while rejecting all other profile changes."""
    existing = comparable_config(existing_config)
    requested = comparable_config(requested_config)
    if existing == requested:
        return True

    existing_rows = existing.pop("selected_rows", None)
    requested_rows = requested.pop("selected_rows", None)
    if not (
        isinstance(existing_rows, int)
        and isinstance(requested_rows, int)
        and requested_rows > existing_rows
    ):
        return False

    # A CIF-enabled input hash covers every selected CIF and therefore changes
    # when the prefix grows. The stable source Parquet hash above must still
    # match, as must the model, prompt, sampling, and engine configuration.
    existing.pop("input_sha256", None)
    requested.pop("input_sha256", None)
    return existing == requested


def prepare_outputs(args: argparse.Namespace, config: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    assert args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, config_path, metrics = (args.output_dir / name for name in ("predictions.jsonl", "run_config.json", "metrics.json"))
    if args.overwrite:
        for path in (predictions, config_path, metrics):
            if path.exists(): path.unlink()
    elif predictions.exists() and not args.resume:
        raise FileExistsError(f"{predictions} exists; use --resume or --overwrite")
    if predictions.exists():
        if not config_path.exists():
            raise ValueError("cannot resume: run_config.json is missing")
        existing_config = json.loads(config_path.read_text())
        if not resume_config_compatible(existing_config, config):
            existing_comparable = comparable_config(existing_config)
            requested_comparable = comparable_config(config)
            differing_fields = sorted(
                key
                for key in existing_comparable.keys() | requested_comparable.keys()
                if existing_comparable.get(key) != requested_comparable.get(key)
            )
            raise ValueError(
                "existing predictions use an incompatible dataset/model/config; "
                f"differing fields: {', '.join(differing_fields)}"
            )
        if comparable_config(existing_config) != comparable_config(config):
            previous_rows = existing_config.get("input", {}).get("selected_rows")
            requested_rows = config.get("input", {}).get("selected_rows")
            atomic_write_json(config_path, config)
            print(
                f"Extending saved run selection from {previous_rows} to "
                f"{requested_rows} rows: {args.output_dir}"
            )
    else:
        atomic_write_json(config_path, config)
    return predictions, config_path, metrics


def read_prediction_records(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not path.exists(): return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip(): continue
            try: record = json.loads(line); idx = int(record["row_index"])
            except Exception as exc: raise ValueError(f"malformed {path}:{line_number}: {exc}") from exc
            if idx in records: raise ValueError(f"duplicate row_index {idx} in {path}")
            records[idx] = record
    return records


def refresh_stored_scores(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Atomically refresh parser-derived fields without changing generations."""
    rows_by_index = {int(row["row_index"]): row for row in rows}
    temporary = path.with_suffix(path.suffix + ".tmp")
    refreshed = 0
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                row_index = int(record["row_index"])
                row = rows_by_index[row_index]
            except Exception as exc:
                raise ValueError(f"cannot refresh {path}:{line_number}: {exc}") from exc
            if str(record.get("id")) != str(row["id"]):
                raise ValueError(f"id mismatch for row_index {row_index}")
            for candidate in record.get("candidates") or []:
                score = score_response(str(candidate.get("text") or ""), row)
                candidate["parsed_formula"] = score["predicted_formula"]
                candidate["parsed_geometry"] = score["parsed_geometry"]
                candidate["scores"] = score
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            refreshed += 1
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    return refreshed


def initialize_vllm(args: argparse.Namespace) -> Tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment") from exc
    kwargs: Dict[str, Any] = {
        "model": args.model, "tokenizer": args.tokenizer or args.model, "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code, "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size, "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb, "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "enable_prefix_caching": args.enable_prefix_caching, "seed": args.seed,
    }
    if "swap_space" in inspect.signature(LLM.__init__).parameters: kwargs["swap_space"] = args.swap_space
    for key in ("revision", "tokenizer_revision", "quantization", "max_model_len", "max_num_seqs", "download_dir"):
        if getattr(args, key) is not None: kwargs[key] = getattr(args, key)
    llm = LLM(**kwargs)
    sampling = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        min_p=args.min_p, repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty, max_tokens=args.max_tokens, skip_special_tokens=True,
        spaces_between_special_tokens=False)
    return llm, sampling


def score_prediction_records(rows: Sequence[Mapping[str, Any]], records: Mapping[int, Mapping[str, Any]]) -> Tuple[list, Dict[str, Any]]:
    scored, prompt_tokens, output_tokens = [], 0, 0
    finish_reasons: Counter[str] = Counter()
    truncations = 0
    for row in rows:
        idx = int(row["row_index"])
        record = records.get(idx)
        if record is None: raise ValueError(f"missing prediction for row_index {idx}")
        if str(record.get("id")) != str(row["id"]): raise ValueError(f"id mismatch for row_index {idx}")
        candidates = record.get("candidates") or []
        if not candidates: raise ValueError(f"no candidates for row_index {idx}")
        scores = []
        for rank, candidate in enumerate(candidates):
            scores.append(score_response(str(candidate.get("text") or ""), row))
            output_tokens += int(candidate.get("output_tokens") or 0)
            reason = str(candidate.get("finish_reason") or "unknown"); finish_reasons[reason] += 1
            truncations += int(rank == 0 and reason == "length")
        prompt_tokens += int(record.get("prompt_tokens") or 0)
        scored.append((row, scores[0], scores))
    return scored, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "finish_reason_counts_all_candidates": dict(finish_reasons),
        "top1_length_truncation_count": truncations,
        "top1_length_truncation_rate": truncations / len(rows) if rows else None}


def generate_pending(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], existing: Mapping[int, Mapping[str, Any]],
                     path: Path, template: Optional[str], llm: Any, sampling: Any, initial_scored: list,
                     checkpoint: Callable[[Sequence[Any], Mapping[str, Any]], None]) -> Dict[str, Any]:
    pending = [row for row in rows if int(row["row_index"]) not in existing]
    started = time.perf_counter(); generated = prompt_tokens = output_tokens = 0
    progress = list(initial_scored); last_save = len(existing)
    def runtime() -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {"resumed_examples": len(existing), "generated_examples_this_run": generated,
            "generation_wall_seconds_this_run": elapsed, "prompt_tokens_this_run": prompt_tokens,
            "output_tokens_this_run": output_tokens, "examples_per_second_this_run": generated / elapsed if elapsed else None,
            "tokens_per_second_this_run": (prompt_tokens + output_tokens) / elapsed if elapsed else None}
    with path.open("a" if path.exists() else "w", encoding="utf-8") as handle:
        begin = 0
        while begin < len(pending):
            completed = len(existing) + generated
            size = min(args.request_chunk_size, args.save_every - completed % args.save_every, len(pending) - begin)
            batch = pending[begin:begin + size]; begin += size
            outputs = llm.chat([
                row.get("_generation_prompt_messages", row["prompt_messages"])
                for row in batch
            ], sampling_params=sampling,
                use_tqdm=not args.quiet_vllm, chat_template=template, add_generation_prompt=True,
                chat_template_kwargs={"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None)
            if len(outputs) != len(batch): raise RuntimeError("vLLM returned the wrong output count")
            for row, request in zip(batch, outputs):
                candidates = []
                for rank, candidate in enumerate(request.outputs):
                    score = score_response(candidate.text, row); tokens = len(candidate.token_ids or [])
                    candidates.append({"rank": rank, "text": candidate.text, "parsed_formula": score["predicted_formula"],
                        "parsed_geometry": score["parsed_geometry"], "finish_reason": getattr(candidate, "finish_reason", None),
                        "stop_reason": getattr(candidate, "stop_reason", None), "output_tokens": tokens, "scores": score})
                    output_tokens += tokens
                prompt_ids = getattr(request, "prompt_token_ids", None) or []; prompt_tokens += len(prompt_ids)
                handle.write(json.dumps({"row_index": int(row["row_index"]), "id": row["id"],
                    "material_id": row["metadata"].get("material_id"), "prompt_tokens": len(prompt_ids),
                    "candidates": candidates}, ensure_ascii=False, separators=(",", ":")) + "\n")
                generated += 1; scores = [candidate["scores"] for candidate in candidates]; progress.append((row, scores[0], scores))
                if (len(existing) + generated) % args.save_every == 0:
                    handle.flush(); os.fsync(handle.fileno()); checkpoint(progress, runtime()); last_save = len(existing) + generated
        if len(existing) + generated != last_save:
            handle.flush(); os.fsync(handle.fileno()); checkpoint(progress, runtime())
    return runtime()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows, profile = load_rows(args.input_parquet, args.start_index, args.max_examples)
    validation = validate_rows(rows); profile["validation"] = validation
    if args.candidate_only_prompt:
        configure_candidate_only_prompts(rows)
    if args.selected_input_prompt:
        configure_selected_input_prompts(rows, profile, args)
    elif args.include_cif:
        configure_cif_prompts(
            rows,
            profile,
            max_cif_chars=args.max_cif_chars,
            layout=args.cif_prompt_layout,
        )
    if args.validate_only:
        print(json.dumps(profile, indent=2, sort_keys=True)); return
    template, source = resolve_chat_template(args, profile)
    config = make_run_config(args, profile, template, source)
    predictions, _, metrics_path = prepare_outputs(args, config)
    existing = read_prediction_records(predictions)
    selected = {int(row["row_index"]) for row in rows}
    extra = set(existing) - selected
    if extra: raise ValueError(f"prediction file has rows outside this selection: {sorted(extra)[:5]}")
    if args.refresh_stored_scores and existing:
        refreshed = refresh_stored_scores(predictions, rows)
        existing = read_prediction_records(predictions)
        print(f"Refreshed parser-derived scores in {refreshed} prediction records: {predictions}")
    completed_rows = [row for row in rows if int(row["row_index"]) in existing]
    initial_scored, resumed_usage = score_prediction_records(completed_rows, existing) if completed_rows else ([], {})
    def checkpoint(scored: Sequence[Any], runtime: Mapping[str, Any]) -> None:
        atomic_write_json(metrics_path, {"status": "complete" if len(scored) == len(rows) else "in_progress",
            "completed_examples": len(scored), "selected_examples": len(rows), "dataset_profile": validation,
            "metrics": summarize_scored(scored), "usage_from_resumed_records": resumed_usage,
            "runtime_this_invocation": dict(runtime), "updated_unix_time": time.time()})
    if len(existing) == len(rows):
        checkpoint(initial_scored, {"resumed_examples": len(existing), "generated_examples_this_run": 0})
        print(f"All {len(rows)} examples already complete. Metrics: {metrics_path}"); return
    llm, sampling = initialize_vllm(args)
    runtime = generate_pending(args, rows, existing, predictions, template, llm, sampling, initial_scored, checkpoint)
    final_records = read_prediction_records(predictions)
    final_scored, usage = score_prediction_records(rows, final_records)
    runtime["final_record_usage"] = usage
    checkpoint(final_scored, runtime)
    print(f"Completed {len(rows)} examples. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
