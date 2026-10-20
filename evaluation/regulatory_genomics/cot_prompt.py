"""Shared CoT prompt transformation for MPRA training and evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence


OUTPUT_INSTRUCTION_RE = re.compile(
    r"Output exactly (\d+) DNA bases and nothing else\.\nDNA Sequence:\s*$"
)
VARIANT_OUTPUT_INSTRUCTION_RE = re.compile(
    r"Return only the variant in this format:\n"
    r"GRCh37\|chromosome\|1-based-position\|reference-allele\|alternate-allele\n"
    r"Variant:\s*$"
)
LOCAL_EDIT_OUTPUT_INSTRUCTION_RE = re.compile(r"Return the edit\.\nEdit:\s*$")
COT_PROMPT_VERSION = 2


def cot_prompt_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Return messages using the exact reasoning format used for CoT SFT."""
    transformed = [dict(message) for message in messages]
    if not transformed:
        raise ValueError("input messages are empty")

    content = str(transformed[-1].get("content") or "")
    if "Return only the local edit." in content or "Return only the edit." in content:
        no_cot_instruction = (
            "Return only the local edit."
            if "Return only the local edit." in content
            else "Return only the edit."
        )
        updated_system_instruction = content.replace(
            no_cot_instruction,
            "Provide reasoning followed by the edit.",
            1,
        )
        replacement = (
            "First provide your reasoning, then return the edit using this format:\n"
            "Reasoning: <reasoning_trace>\n"
            "Edit: <edit>"
        )
        updated, substitutions = LOCAL_EDIT_OUTPUT_INSTRUCTION_RE.subn(
            replacement, updated_system_instruction
        )
        expected = "local-edit"
    elif "Return only the variant record." in content:
        updated_system_instruction = content.replace(
            "Return only the variant record.",
            "Provide reasoning followed by the variant record.",
            1,
        )
        replacement = (
            "First provide your reasoning, then output the variant using this format:\n"
            "Reasoning: <reasoning_trace>\n"
            "Variant: GRCh37|chromosome|1-based-position|reference-allele|alternate-allele"
        )
        updated, substitutions = VARIANT_OUTPUT_INSTRUCTION_RE.subn(
            replacement, updated_system_instruction
        )
        expected = "variant-record"
    else:
        updated_system_instruction = content.replace(
            "Return only the DNA sequence.",
            "Provide reasoning followed by the DNA sequence.",
            1,
        )
        if updated_system_instruction == content:
            raise ValueError("prompt does not contain a recognized output instruction")

        replacement = (
            "First provide your reasoning, then output exactly \\1 DNA bases using this format:\n"
            "Reasoning: <reasoning_traces>\n"
            "DNA Sequence: <sequence>"
        )
        updated, substitutions = OUTPUT_INSTRUCTION_RE.subn(
            replacement, updated_system_instruction
        )
        expected = "DNA"
    if substitutions != 1:
        raise ValueError(f"prompt does not end with the expected {expected} output instruction")

    transformed[-1]["content"] = updated
    return transformed
