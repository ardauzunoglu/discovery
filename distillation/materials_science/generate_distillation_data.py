#!/usr/bin/env python3
"""Generate teacher reasoning traces for supervised distillation.

Each teacher request contains a cleaned set of desired material characteristics
and its known-good output. Original task instructions and response templates
are not sent to the teacher. The teacher is asked to explain how to derive that
fixed output, not to solve the task again. Results are appended to a resumable
JSONL file and include separate ``input``/``output``/``reasoning`` fields and
ready-to-use chat ``messages``. Other source fields are carried through
unchanged, allowing task metadata and verifier columns to survive the pass.

The input may be a JSONL file or a Parquet file.  By default, examples are read
from a chat ``messages`` field whose final message is an assistant answer.  A
dataset with explicit ``input`` and ``output`` fields is also supported.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TEACHER_SYSTEM_PROMPT = """You create high-quality supervised distillation data.
You will receive desired material characteristics, a chemical system, and a
known correct output. Write a rigorous, self-contained reasoning trace that
connects every supplied characteristic to the known output. Treat that output
as fixed: do not revise, contradict, or critique it. Do not mention that the
answer was supplied to you, and do not repeat the final answer. Return only the
reasoning trace enclosed in <reasoning> and </reasoning> tags."""

DISTILLATION_INPUT_INSTRUCTION = """Derive the candidate material from the
chemical system and desired material characteristics below. Give a clear
reasoning trace that connects each characteristic to the candidate."""

MATERIAL_CONSTRAINT_FIELDS = (
    "Chemical system",
    "Material family",
    "Stability",
    "Electronic character",
    "Formation energy",
)

REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
REASONING_LABEL_RE = re.compile(r"(?im)^\s*Reasoning\s*:")
ANSWER_LABEL_RE = re.compile(
    r"(?im)^\s*(?:Candidate|Final answer|Answer|Output)\s*:"
)


def infer_provider(model: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    name = model.lower().split("/")[-1]
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Cannot infer a provider from model {model!r}; pass --provider explicitly"
    )


def default_api_key_env(provider: str) -> str:
    if provider == "gateway":
        return "GATEWAY_KEY"
    return "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"


def load_api_helpers() -> Any:
    """Load the API client shared with the evaluators without requiring PYTHONPATH setup."""
    helper_dir = Path(__file__).resolve().parents[2] / "evaluation" / "materials_science"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    import evaluate_matsci_api

    return evaluate_matsci_api


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ProgressTracker:
    """Small progress display for terminals and batch logs."""

    def __init__(self, total: int, initial: int = 0, enabled: bool = True) -> None:
        self.total = total
        self.initial = initial
        self.completed = initial
        self.failed = 0
        self.enabled = enabled
        self.started = time.perf_counter()
        self.last_reported = initial
        self.last_report_time = self.started
        self.is_tty = sys.stderr.isatty()
        self.closed = False
        self._report(force=True)

    def advance(self, success: bool = True) -> None:
        if success:
            self.completed += 1
        else:
            self.failed += 1
        self._report(force=not success or self.completed >= self.total)

    def _report(self, force: bool = False) -> None:
        if not self.enabled or self.closed:
            return
        now = time.perf_counter()
        report_step = max(1, self.total // 100)
        if not (
            force or self.is_tty or self.completed - self.last_reported >= report_step
            or now - self.last_report_time >= 30
        ):
            return
        elapsed = now - self.started
        generated = self.completed - self.initial
        rate = generated / elapsed if generated and elapsed else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate else None
        failure = f" | failed {self.failed}" if self.failed else ""
        line = (
            f"Progress {self.completed}/{self.total} "
            f"({100 * self.completed / self.total:5.1f}%) | {rate:.2f} ex/s | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}{failure}"
        )
        sys.stderr.write("\r" + line if self.is_tty else line + "\n")
        sys.stderr.flush()
        self.last_reported = self.completed
        self.last_report_time = now

    def close(self) -> None:
        if not self.enabled or self.closed:
            return
        if self.last_reported != self.completed:
            self._report(force=True)
        if self.is_tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.closed = True


def parse_args(
    argv: Optional[Sequence[str]] = None,
    configure_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces conditioned on each input and known output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl or .parquet dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Resumable output JSONL path.")
    parser.add_argument("--provider", choices=("openai", "anthropic", "gateway"))
    parser.add_argument("--model", help="Teacher model name.")
    parser.add_argument("--input-field", help="Explicit input field (disables chat-schema inference).")
    parser.add_argument("--output-field", help="Explicit known-output field.")
    parser.add_argument("--id-field", default="id")

    data = parser.add_argument_group("data selection")
    data.add_argument("--start-index", type=int, default=0)
    data.add_argument("--max-examples", type=int)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--max-tokens", type=int, default=4096)
    generation.add_argument("--temperature", type=float)
    generation.add_argument("--top-p", type=float)
    generation.add_argument("--reasoning-effort")
    generation.add_argument(
        "--system-prompt",
        default=TEACHER_SYSTEM_PROMPT,
        help="Teacher instruction. It should require <reasoning> tags.",
    )

    client = parser.add_argument_group("API client")
    client.add_argument("--api-key-env")
    client.add_argument("--base-url")
    client.add_argument("--max-concurrency", type=int, default=5)
    client.add_argument("--request-chunk-size", type=int, default=64)
    client.add_argument("--timeout", type=float, default=600.0)
    client.add_argument("--max-retries", type=int, default=5)

    output = parser.add_argument_group("output/recovery")
    output.add_argument("--save-every", type=int, default=25)
    output.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument(
        "--skip-failed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Continue after per-example generation failures instead of stopping "
            "the entire run. Failed rows are reported to stderr and omitted from "
            "the output JSONL."
        ),
    )
    output.add_argument(
        "--allow-max-tokens-increase-on-resume",
        action="store_true",
        help=(
            "Allow an existing compatible output to resume with a strictly larger "
            "--max-tokens ceiling. The change is recorded in the run config."
        ),
    )
    output.add_argument("--quiet", action="store_true")
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first teacher request and exit without calling an API or writing output.",
    )
    output.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and count selected examples without calling an API or writing output.",
    )

    if configure_parser is not None:
        configure_parser(parser)

    args = parser.parse_args(argv)
    if args.start_index < 0 or (args.max_examples is not None and args.max_examples <= 0):
        parser.error("indices/counts must be positive (start-index may be zero)")
    if min(args.max_tokens, args.max_concurrency, args.request_chunk_size, args.save_every) <= 0:
        parser.error("generation, concurrency, chunk, and save values must be positive")
    if args.timeout <= 0 or args.max_retries < 0:
        parser.error("--timeout must be positive and --max-retries may not be negative")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.top_p is not None and not 0 <= args.top_p <= 1:
        parser.error("--top-p must be between 0 and 1")
    if bool(args.input_field) != bool(args.output_field):
        parser.error("--input-field and --output-field must be provided together")
    if not (args.dry_run or args.validate_only) and not args.model:
        parser.error("--model is required unless --dry-run or --validate-only is used")
    if args.overwrite:
        args.resume = False
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _normalize_messages(value: Any, row_index: int) -> List[Dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"row {row_index}: messages must be a non-empty list")
    messages: List[Dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError(f"row {row_index}: every message must be an object")
        role = str(message.get("role") or "")
        if role not in ("system", "developer", "user", "assistant"):
            raise ValueError(f"row {row_index}: unsupported message role {role!r}")
        messages.append({"role": role, "content": _text(message.get("content"))})
    return messages


def extract_example(
    row: Mapping[str, Any],
    row_index: int,
    input_field: Optional[str] = None,
    output_field: Optional[str] = None,
    id_field: str = "id",
) -> Dict[str, Any]:
    """Normalize either a chat example or an explicit input/output pair."""
    example_id = _text(row.get(id_field)) or str(row_index)
    if input_field and output_field:
        if input_field not in row or output_field not in row:
            missing = [name for name in (input_field, output_field) if name not in row]
            raise ValueError(f"row {row_index}: missing field(s): {', '.join(missing)}")
        input_value = row[input_field]
        if isinstance(input_value, list):
            input_messages = _normalize_messages(input_value, row_index)
        else:
            input_messages = [{"role": "user", "content": _text(input_value)}]
        known_output = _text(row[output_field])
    elif "messages" in row and row.get("messages") is not None:
        messages = _normalize_messages(row["messages"], row_index)
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                f"row {row_index}: final messages entry must be the known assistant output"
            )
        input_messages, known_output = messages[:-1], messages[-1]["content"]
    elif "input" in row and "output" in row:
        input_value = row["input"]
        input_messages = (
            _normalize_messages(input_value, row_index)
            if isinstance(input_value, list)
            else [{"role": "user", "content": _text(input_value)}]
        )
        known_output = _text(row["output"])
    else:
        raise ValueError(
            f"row {row_index}: expected chat 'messages' or explicit 'input'/'output' fields"
        )
    if not input_messages:
        raise ValueError(f"row {row_index}: input has no messages")
    if not known_output.strip():
        raise ValueError(f"row {row_index}: known output is empty")
    return {
        "row_index": row_index,
        "id": example_id,
        "input": input_messages,
        "output": known_output,
        "_source_record": dict(row),
    }


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {index}: JSONL value is not an object")
            yield index, value


def _iter_json(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else [value]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: JSON value is not an object")
        yield index, row


def _iter_parquet(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet requires pyarrow") from exc
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def load_examples(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = args.input
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        iterator = _iter_jsonl(path)
    elif suffix == ".json":
        iterator = _iter_json(path)
    elif suffix in (".parquet", ".pq"):
        iterator = _iter_parquet(path)
    else:
        raise ValueError("--input must be a .jsonl, .json, .parquet, or .pq file")

    examples: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    stop = args.start_index + args.max_examples if args.max_examples else None
    total_seen = 0
    for row_index, row in iterator:
        if row_index < args.start_index:
            total_seen = row_index + 1
            continue
        if stop is not None and row_index >= stop:
            break
        total_seen = row_index + 1
        example = extract_example(
            row, row_index, args.input_field, args.output_field, args.id_field,
        )
        if example["id"] in seen_ids:
            raise ValueError(f"duplicate selected id {example['id']!r}")
        seen_ids.add(example["id"])
        examples.append(example)
    if not examples:
        raise ValueError("data selection contains no examples")
    return examples, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "selected_start_index": args.start_index,
        "selected_rows": len(examples),
        "rows_scanned": total_seen,
    }


def extract_material_constraints(messages: Sequence[Mapping[str, str]]) -> str:
    """Keep only the labeled material data and discard all source instructions."""
    values: Dict[str, str] = {}
    labels = {label.casefold(): label for label in MATERIAL_CONSTRAINT_FIELDS}
    for message in messages:
        for line in str(message.get("content") or "").splitlines():
            if ":" not in line:
                continue
            raw_label, raw_value = line.split(":", 1)
            label = labels.get(raw_label.strip().casefold())
            value = raw_value.strip()
            if label and value:
                values[label] = value
    if "Chemical system" not in values:
        raise ValueError("input does not contain a labeled 'Chemical system' constraint")
    missing = [label for label in MATERIAL_CONSTRAINT_FIELDS if label not in values]
    if missing:
        raise ValueError(f"input is missing material constraint(s): {', '.join(missing)}")
    return "\n".join(f"{label}: {values[label]}" for label in MATERIAL_CONSTRAINT_FIELDS)


def clean_known_output(known_output: str) -> str:
    """Remove an existing rationale while retaining the fixed structured answer."""
    answer = known_output.strip()
    reasoning_label = REASONING_LABEL_RE.search(answer)
    if reasoning_label:
        answer_label = ANSWER_LABEL_RE.search(answer, reasoning_label.end())
        if answer_label:
            answer = answer[answer_label.start():].lstrip()
    return answer


def make_distillation_input(constraints: str) -> str:
    return f"{DISTILLATION_INPUT_INSTRUCTION}\n\n{constraints}"


def make_teacher_messages(
    example: Mapping[str, Any], system_prompt: str = TEACHER_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    constraints = extract_material_constraints(example["input"])
    known_output = clean_known_output(str(example["output"]))
    user = (
        "Create the reasoning trace for this distillation example. Use only the "
        "supplied material data. Return your output enclosed in <reasoning> and "
        "</reasoning> tags.\n\n"
        "<desired_material_characteristics>\n"
        f"{constraints}\n"
        "</desired_material_characteristics>\n\n"
        "<known_output>\n"
        f"{known_output}\n"
        "</known_output>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def parse_reasoning(text: str) -> str:
    match = REASONING_RE.search(text)
    if not match:
        raise ValueError("teacher response does not contain <reasoning>...</reasoning>")
    reasoning = match.group(1).strip()
    if not reasoning:
        raise ValueError("teacher returned an empty reasoning trace")
    return reasoning


def distilled_assistant_message(reasoning: str, known_output: str) -> str:
    """Insert a teacher trace while preserving the known structured answer."""
    answer = clean_known_output(known_output)
    return f"Reasoning: {reasoning.strip()}\n{answer}"


def read_completed(path: Path) -> Dict[int, str]:
    completed: Dict[int, str] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["row_index"])
                record_id = str(record["id"])
            except Exception as exc:
                raise ValueError(f"invalid output record on line {line_number}: {exc}") from exc
            if index in completed:
                raise ValueError(f"duplicate row_index {index} in {path}")
            completed[index] = record_id
    return completed


def config_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".config.json")


def make_config(args: argparse.Namespace, provider: str, profile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "input": dict(profile),
        "provider": provider,
        "model": args.model,
        "api_key_env": args.api_key_env or default_api_key_env(provider),
        "base_url": args.base_url,
        "fields": {
            "input": args.input_field or "auto(messages|input)",
            "output": args.output_field or "auto(final assistant|output)",
            "id": args.id_field,
        },
        "input_transform": {
            "type": "labeled_material_constraints_v1",
            "retained_fields": list(MATERIAL_CONSTRAINT_FIELDS),
            "distillation_instruction": DISTILLATION_INPUT_INSTRUCTION,
            "remove_existing_output_reasoning": True,
        },
        "teacher_system_prompt": args.system_prompt,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reasoning_effort": args.reasoning_effort,
        },
        "client": {
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "max_concurrency": args.max_concurrency,
            "request_chunk_size": args.request_chunk_size,
        },
    }


def comparable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    input_config = config.get("input", {})
    return {
        "input_sha256": input_config.get("sha256"),
        "selected_start_index": input_config.get("selected_start_index"),
        "selected_rows": input_config.get("selected_rows"),
        **{key: config.get(key) for key in (
            "provider", "model", "base_url", "fields", "input_transform",
            "teacher_system_prompt", "sampling",
        )},
    }


def configs_differ_only_by_max_tokens_increase(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> tuple[bool, Optional[int], Optional[int]]:
    """Return whether NEW differs only by a strictly larger max-token ceiling."""
    old_comparable = comparable_config(old)
    new_comparable = comparable_config(new)
    old_sampling = dict(old_comparable.get("sampling") or {})
    new_sampling = dict(new_comparable.get("sampling") or {})
    old_max = old_sampling.pop("max_tokens", None)
    new_max = new_sampling.pop("max_tokens", None)
    old_comparable["sampling"] = old_sampling
    new_comparable["sampling"] = new_sampling
    valid_limits = (
        isinstance(old_max, int)
        and not isinstance(old_max, bool)
        and isinstance(new_max, int)
        and not isinstance(new_max, bool)
        and new_max > old_max
    )
    return valid_limits and old_comparable == new_comparable, old_max, new_max


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_output(
    args: argparse.Namespace, config: Mapping[str, Any], examples: Sequence[Mapping[str, Any]],
) -> Dict[int, str]:
    output, metadata = args.output, config_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (output, metadata):
            if path.exists():
                path.unlink()
    elif output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; use --resume or --overwrite")

    completed = read_completed(output)
    if output.exists():
        if not metadata.exists():
            raise ValueError(f"cannot resume: missing {metadata}")
        old = json.loads(metadata.read_text(encoding="utf-8"))
        if comparable_config(old) != comparable_config(config):
            allowed, old_max, new_max = configs_differ_only_by_max_tokens_increase(
                old, config
            )
            if not (args.allow_max_tokens_increase_on_resume and allowed):
                raise ValueError("existing output uses an incompatible input/model/config")
            updated_config = dict(config)
            changes = list(old.get("resume_changes") or [])
            changes.append({
                "field": "sampling.max_tokens",
                "from": old_max,
                "to": new_max,
                "completed_rows_before_change": len(completed),
            })
            updated_config["resume_changes"] = changes
            atomic_write_json(metadata, updated_config)
            if not args.quiet:
                print(
                    f"Resuming with increased max_tokens: {old_max} -> {new_max}; "
                    f"retaining {len(completed)} completed example(s).",
                    file=sys.stderr,
                )
    else:
        atomic_write_json(metadata, config)

    expected = {int(example["row_index"]): str(example["id"]) for example in examples}
    extra = sorted(set(completed) - set(expected))
    if extra:
        raise ValueError(f"output contains rows outside this selection: {extra[:5]}")
    for index, record_id in completed.items():
        if record_id != expected[index]:
            raise ValueError(f"id mismatch for row_index {index}")
    if completed and not args.quiet:
        print(
            f"Resuming {output}: found {len(completed)}/{len(examples)} completed "
            "example(s); they will be skipped.",
            file=sys.stderr,
        )
    return completed


async def generate_one(
    provider: str,
    client: Any,
    args: argparse.Namespace,
    example: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
    teacher_message_builder: Optional[
        Callable[[Mapping[str, Any], str], List[Dict[str, str]]]
    ] = None,
    record_builder: Optional[
        Callable[[str, argparse.Namespace, Mapping[str, Any], Mapping[str, Any], str], Dict[str, Any]]
    ] = None,
    reasoning_validator: Optional[
        Callable[[Mapping[str, Any], str], None]
    ] = None,
) -> Dict[str, Any]:
    # Imported only for actual generation so --help, --dry-run, and validation
    # remain usable before API dependencies are installed.
    api = load_api_helpers()
    request = {
        "openai": api.request_openai,
        "anthropic": api.request_anthropic,
        "gateway": api.request_gateway,
    }[provider]
    original_teacher_messages = (teacher_message_builder or make_teacher_messages)(
        example, args.system_prompt
    )
    quality_retries = int(getattr(args, "max_quality_retries", 0))
    validation_error: Optional[ValueError] = None
    async with semaphore:
        for attempt in range(quality_retries + 1):
            teacher_messages = [dict(message) for message in original_teacher_messages]
            if validation_error is not None:
                correction = (
                    "\n\nYour previous draft was rejected by the distillation quality "
                    f"gate: {validation_error}. Rewrite it from scratch and obey the "
                    "privileged-context policy exactly."
                )
                teacher_messages[-1]["content"] = (
                    str(teacher_messages[-1].get("content") or "") + correction
                )
            result = await request(client, args, teacher_messages)
            if result.get("finish_reason") == "length":
                raise ValueError("teacher response was truncated; increase --max-tokens")
            try:
                reasoning = parse_reasoning(str(result.get("text") or ""))
                if reasoning_validator is not None:
                    reasoning_validator(example, reasoning)
            except ValueError as exc:
                validation_error = exc
                if attempt >= quality_retries:
                    raise ValueError(
                        "teacher response failed the distillation quality gate after "
                        f"{quality_retries + 1} attempt(s): {exc}"
                    ) from exc
                continue
            break
    if record_builder is not None:
        return record_builder(provider, args, example, result, reasoning)
    constraints = extract_material_constraints(example["input"])
    known_output = clean_known_output(str(example["output"]))
    messages = [{"role": "user", "content": make_distillation_input(constraints)}]
    messages.append({
        "role": "assistant",
        "content": distilled_assistant_message(reasoning, known_output),
    })
    record = dict(example.get("_source_record") or {})
    if args.input_field:
        record[args.input_field] = constraints
    if args.output_field:
        record[args.output_field] = known_output
    record.update({
        "row_index": int(example["row_index"]),
        "id": example["id"],
        "input": constraints,
        "output": known_output,
        "reasoning": reasoning,
        "messages": messages,
        "prompt_messages": messages[:-1],
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
    })
    return record


async def generate_all(
    provider: str,
    client: Any,
    args: argparse.Namespace,
    examples: Sequence[Mapping[str, Any]],
    completed: Mapping[int, str],
    teacher_message_builder: Optional[
        Callable[[Mapping[str, Any], str], List[Dict[str, str]]]
    ] = None,
    record_builder: Optional[
        Callable[[str, argparse.Namespace, Mapping[str, Any], Mapping[str, Any], str], Dict[str, Any]]
    ] = None,
    reasoning_validator: Optional[
        Callable[[Mapping[str, Any], str], None]
    ] = None,
) -> None:
    pending = [e for e in examples if int(e["row_index"]) not in completed]
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tracker = ProgressTracker(len(examples), len(completed), enabled=not args.quiet)
    generated = 0
    skipped = 0
    mode = "a" if args.output.exists() else "w"

    async def guarded(example: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Any]:
        try:
            return example, await generate_one(
                provider,
                client,
                args,
                example,
                semaphore,
                teacher_message_builder=teacher_message_builder,
                record_builder=record_builder,
                reasoning_validator=reasoning_validator,
            )
        except Exception as exc:
            return example, exc

    try:
        with args.output.open(mode, encoding="utf-8") as handle:
            for begin in range(0, len(pending), args.request_chunk_size):
                batch = pending[begin:begin + args.request_chunk_size]
                tasks = [asyncio.create_task(guarded(example)) for example in batch]
                first_error: Optional[BaseException] = None
                for task in asyncio.as_completed(tasks):
                    example, result = await task
                    if isinstance(result, BaseException):
                        row_error = RuntimeError(
                            f"row {example['row_index']} ({example['id']}): {result}"
                        )
                        if args.skip_failed:
                            skipped += 1
                            sys.stderr.write(f"\nSkipping {row_error}\n")
                            sys.stderr.flush()
                        else:
                            first_error = first_error or row_error
                        tracker.advance(success=False)
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    generated += 1
                    if generated % args.save_every == 0:
                        handle.flush()
                        os.fsync(handle.fileno())
                    tracker.advance()
                handle.flush()
                os.fsync(handle.fileno())
                if first_error is not None:
                    raise first_error
    finally:
        tracker.close()
    if skipped:
        sys.stderr.write(
            f"Skipped {skipped} failed example(s); successful records were kept "
            f"in {args.output}.\n"
        )
        sys.stderr.flush()


async def async_main(
    args: argparse.Namespace, examples: Sequence[Mapping[str, Any]], profile: Mapping[str, Any],
) -> None:
    provider = infer_provider(str(args.model), args.provider)
    if provider == "gateway" and "/" not in str(args.model):
        raise ValueError("gateway models must be provider-prefixed, for example openai/gpt-5.2")
    config = make_config(args, provider, profile)
    completed = prepare_output(args, config, examples)
    if len(completed) == len(examples):
        print(f"All {len(examples)} examples already complete: {args.output}")
        return
    client = load_api_helpers().create_client(provider, args)
    try:
        await generate_all(provider, client, args, examples, completed)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
    final_count = len(read_completed(args.output))
    print(f"Completed {final_count}/{len(examples)} examples: {args.output}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    examples, profile = load_examples(args)
    if args.validate_only:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return
    if args.dry_run:
        print(json.dumps(make_teacher_messages(examples[0], args.system_prompt), indent=2))
        return
    asyncio.run(async_main(args, examples, profile))


if __name__ == "__main__":
    main()
