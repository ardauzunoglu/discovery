#!/usr/bin/env python3
"""Generate supervised reasoning traces for MPRA sequence design.

The teacher sees the original design request and the known alternate sequence,
then infers and explains their sequence differences. Output records preserve the source
columns and add evaluation-aligned chat messages containing both the reasoning
trace and the known DNA answer. API transport and resumable JSONL handling are
shared with the materials distillation generator.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for helper_dir in (
    REPO_ROOT / "distillation" / "materials_science",
    REPO_ROOT / "evaluation" / "regulatory_genomics",
):
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

from cot_prompt import cot_prompt_messages

_BASE_PATH = REPO_ROOT / "distillation" / "materials_science" / "generate_distillation_data.py"
_BASE_SPEC = importlib.util.spec_from_file_location("matsci_distillation_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load shared distillation generator: {_BASE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)

TEACHER_SYSTEM_PROMPT = """You create high-quality supervised distillation data for MPRA sequence design.
You will receive a design request, a privileged target sequence, and a
privileged edit annotation. Silently use the privileged data only to verify the
result of your reasoning and keep the edit coordinate accurate. The written trace must
read as a forward solution from the reference sequence and requested activity
changes: analyze the local regulatory context, identify and justify a specific
edit, and independently arrive at the target edit.

Do not frame the trace as explaining, rationalizing, checking, or reverse-
engineering a known answer. Do not mention the target, privileged context, or
that an answer was supplied. Do not begin by comparing the reference with the
target. Address K562, HepG2, and SK-N-SH, while avoiding claims of experimental
certainty beyond the requested activity values. Do not repeat the complete
alternate sequence or add a final-answer section. Return only the reasoning
trace enclosed in <reasoning> and </reasoning> tags."""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Generate teacher reasoning traces for MPRA sequence-design data."
    parser.set_defaults(
        input_field="prompt_messages",
        output_field="alternate_sequence",
        id_field="pair_id",
        system_prompt=TEACHER_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--max-quality-retries",
        type=int,
        default=2,
        help="Retries when a trace exposes or retrospectively explains the privileged target.",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = base.parse_args(argv, configure_parser=configure_parser)
    if args.max_quality_retries < 0:
        raise ValueError("--max-quality-retries may not be negative")
    return args


def transformed_prompt_messages(example: Mapping[str, Any]) -> List[Dict[str, str]]:
    return cot_prompt_messages(example["input"])


def privileged_edit_annotation(example: Mapping[str, Any]) -> str:
    source = example.get("_source_record") or {}
    reference = str(source.get("reference_sequence") or "").upper()
    alternate = str(example["output"]).strip().upper()
    if not reference:
        raise ValueError("source record has no reference_sequence")

    prefix = 0
    common = min(len(reference), len(alternate))
    while prefix < common and reference[prefix] == alternate[prefix]:
        prefix += 1
    suffix = 0
    max_suffix = min(len(reference) - prefix, len(alternate) - prefix)
    while suffix < max_suffix and reference[-1 - suffix] == alternate[-1 - suffix]:
        suffix += 1
    ref_end = len(reference) - suffix
    alt_end = len(alternate) - suffix
    removed = reference[prefix:ref_end]
    inserted = alternate[prefix:alt_end]

    if len(removed) == len(inserted) == 1:
        return f"Position {prefix + 1} (one-based); substitution {removed} -> {inserted}."
    if not removed and inserted:
        return (
            f"Insert {inserted} between positions {prefix} and {prefix + 1} "
            "(one-based flanking coordinates)."
        )
    if removed and not inserted:
        end = prefix + len(removed)
        return f"Delete {removed} at positions {prefix + 1}-{end} (one-based)."
    if removed or inserted:
        end = prefix + len(removed)
        return (
            f"Replace positions {prefix + 1}-{end} (one-based), {removed or '<empty>'} "
            f"-> {inserted or '<empty>'}."
        )
    raise ValueError("reference and alternate sequences are identical")


def make_teacher_messages(
    example: Mapping[str, Any], system_prompt: str = TEACHER_SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    prompt = transformed_prompt_messages(example)[-1]["content"]
    known_sequence = str(example["output"]).strip().upper()
    if not known_sequence or set(known_sequence) - set("ACGTN"):
        raise ValueError("known alternate sequence is empty or contains non-DNA characters")
    edit_annotation = privileged_edit_annotation(example)
    user = (
        "Solve this MPRA sequence-design problem from first principles. Produce a "
        "forward reasoning trace that selects and justifies the edit; return only "
        "that trace inside <reasoning> tags. The privileged target is for silent "
        "verification and must not be referenced in the trace.\n\n"
        "<design_request>\n"
        f"{prompt}\n"
        "</design_request>\n\n"
        "<privileged_target_sequence>\n"
        f"{known_sequence}\n"
        "</privileged_target_sequence>\n\n"
        "<privileged_edit_annotation>\n"
        f"{edit_annotation}\n"
        "</privileged_edit_annotation>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def validate_reasoning(example: Mapping[str, Any], reasoning: str) -> None:
    known_sequence = str(example["output"]).strip().upper()
    compact = "".join(reasoning.upper().split())
    if known_sequence in compact:
        raise ValueError("reasoning repeats the complete known alternate sequence")
    retrospective = re.compile(
        r"\b(?:known|given|supplied|provided|fixed|privileged)\s+"
        r"(?:answer|target|alternate(?:\s+sequence)?|sequence)\b|"
        r"\breverse[- ]engineer(?:ed|ing)?\b|"
        r"\bcompar(?:e|ing)\s+the\s+reference\s+(?:and|to|with)\s+the\s+alternate\b",
        re.IGNORECASE,
    )
    match = retrospective.search(reasoning)
    if match:
        raise ValueError(f"reasoning exposes retrospective target framing: {match.group(0)!r}")


def build_record(
    provider: str,
    args: argparse.Namespace,
    example: Mapping[str, Any],
    result: Mapping[str, Any],
    reasoning: str,
) -> Dict[str, Any]:
    known_sequence = str(example["output"]).strip().upper()
    prompt_messages = transformed_prompt_messages(example)
    assistant = f"Reasoning: {reasoning.strip()}\nDNA Sequence: {known_sequence}"
    record = dict(example.get("_source_record") or {})
    record.update(
        {
            "row_index": int(example["row_index"]),
            "id": example["id"],
            "prompt": prompt_messages[-1]["content"],
            "completion": assistant,
            "input": prompt_messages,
            "output": known_sequence,
            "reasoning": reasoning.strip(),
            "prompt_messages": prompt_messages,
            "messages": [*prompt_messages, {"role": "assistant", "content": assistant}],
            "teacher": {
                "provider": provider,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "finish_reason": result.get("finish_reason"),
                "stop_reason": result.get("stop_reason"),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "request_id": result.get("request_id"),
                "response_id": result.get("response_id"),
            },
        }
    )
    return record


def make_config(
    args: argparse.Namespace, provider: str, profile: Mapping[str, Any]
) -> Dict[str, Any]:
    config = base.make_config(args, provider, profile)
    config["input_transform"] = {
        "type": "mpra_forward_derived_reasoning_v4",
        "privileged_target_policy": "silent verification with deterministic edit annotation",
        "written_trace": "derive edit forward from reference and requested activities",
        "response_format": "Reasoning: <reasoning_traces>\\nDNA Sequence: <sequence>",
        "preserve_source_fields": True,
    }
    return config


async def async_main(
    args: argparse.Namespace,
    examples: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> None:
    provider = base.infer_provider(str(args.model), args.provider)
    if provider == "gateway" and "/" not in str(args.model):
        raise ValueError(
            "gateway models must be provider-prefixed, for example anthropic/claude-opus-5"
        )
    config = make_config(args, provider, profile)
    completed = base.prepare_output(args, config, examples)
    if len(completed) == len(examples):
        print(f"All {len(examples)} examples already complete: {args.output}")
        return
    client = base.load_api_helpers().create_client(provider, args)
    try:
        await base.generate_all(
            provider,
            client,
            args,
            examples,
            completed,
            teacher_message_builder=make_teacher_messages,
            record_builder=build_record,
            reasoning_validator=validate_reasoning,
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            value = close()
            if asyncio.iscoroutine(value):
                await value
    final_count = len(base.read_completed(args.output))
    print(f"Completed {final_count}/{len(examples)} examples: {args.output}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    examples, profile = base.load_examples(args)
    if args.validate_only:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return
    if args.dry_run:
        print(json.dumps(make_teacher_messages(examples[0], args.system_prompt), indent=2))
        return
    asyncio.run(async_main(args, examples, profile))


if __name__ == "__main__":
    main()
