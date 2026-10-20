#!/usr/bin/env python3
"""Evaluate OpenAI GPT or Anthropic Claude API models on the MP temporal split.

The output and metrics schema intentionally matches ``evaluate_matsci_vllm.py``.
Requests are concurrent, while completed rows are immediately appended to a
resumable JSONL file.  Provider SDKs are imported lazily so
validation, tests, and metric recomputation do not require API dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_matsci_vllm import (
    atomic_write_json,
    load_rows,
    read_prediction_records,
    score_response,
    summarize_scored,
    validate_rows,
)


TOKENS_PER_MILLION = 1_000_000


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ProgressTracker:
    """Small dependency-free progress display suitable for terminals and job logs."""

    def __init__(
        self, total: int, initial: int = 0, enabled: bool = True, stream: Any = sys.stderr,
        details: str = "",
    ) -> None:
        self.total = total
        self.initial = initial
        self.completed = initial
        self.failed = 0
        self.details = details
        self.enabled = enabled
        self.stream = stream
        self.started = time.perf_counter()
        self.last_reported = initial
        self.last_reported_failures = 0
        self.last_report_time = self.started
        self.last_width = 0
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.closed = False
        self._report(force=True)

    def advance(self, success: bool = True, details: Optional[str] = None) -> None:
        if details is not None:
            self.details = details
        if success:
            self.completed += 1
        else:
            self.failed += 1
        self._report(force=self.completed >= self.total or not success)

    def _line(self, now: float) -> str:
        elapsed = now - self.started
        generated = self.completed - self.initial
        rate = generated / elapsed if generated and elapsed else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate else None
        fraction = self.completed / self.total if self.total else 1.0
        filled = min(20, round(20 * fraction))
        details = f" | {self.details}" if self.details else ""
        failure = f" | failed {self.failed}" if self.failed else ""
        return (
            f"Progress [{'#' * filled}{'-' * (20 - filled)}] "
            f"{self.completed}/{self.total} ({100 * fraction:5.1f}%) | "
            f"{rate:.2f} ex/s | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
            f"{details}"
            f"{failure}"
        )

    def _report(self, force: bool = False) -> None:
        if not self.enabled or self.closed:
            return
        now = time.perf_counter()
        report_step = max(1, self.total // 100)
        if not (
            force
            or self.is_tty
            or self.completed - self.last_reported >= report_step
            or now - self.last_report_time >= 30
        ):
            return
        line = self._line(now)
        if self.is_tty:
            self.stream.write("\r" + line.ljust(self.last_width))
            self.last_width = max(self.last_width, len(line))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_reported = self.completed
        self.last_reported_failures = self.failed
        self.last_report_time = now

    def close(self) -> None:
        if not self.enabled or self.closed:
            return
        if self.last_reported != self.completed or self.last_reported_failures != self.failed:
            self._report(force=True)
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


class ProviderRefusalError(RuntimeError):
    """A provider refusal that should not be saved as a completed prediction."""

    def __init__(self, row: Mapping[str, Any], stop_details: Any) -> None:
        self.row_index = int(row["row_index"])
        self.example_id = str(row["id"])
        self.stop_details = _plain_json(stop_details)
        details = json.dumps(self.stop_details, ensure_ascii=False, sort_keys=True)
        super().__init__(
            f"row {self.row_index} ({self.example_id}): provider refused the "
            f"request; stop_details={details}"
        )

    def as_record(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "id": self.example_id,
            "error_type": "provider_refusal",
            "stop_details": self.stop_details,
        }


def parse_args(
    argv: Optional[Sequence[str]] = None,
    configure_parser: Optional[Any] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI GPT or Anthropic Claude API model on the materials temporal split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic", "gateway"))
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")

    data = parser.add_argument_group("data selection")
    data.add_argument("--start-index", type=int, default=0)
    data.add_argument("--max-examples", type=int)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--max-tokens", type=int, default=8192)
    generation.add_argument(
        "--temperature", type=float,
        help="Send a sampling temperature. Omitted by default for broad model compatibility.",
    )
    generation.add_argument(
        "--top-p", type=float,
        help="Send nucleus sampling. Omitted by default for broad model compatibility.",
    )
    generation.add_argument(
        "--reasoning-effort",
        help="OpenAI reasoning effort or Anthropic output effort, for models that support it.",
    )
    generation.add_argument(
        "--generations-per-input", "--n", dest="n", type=int, default=1,
        help="Independent API calls (candidate generations) per input.",
    )

    pricing = parser.add_argument_group("cost tracking")
    pricing.add_argument(
        "--input-price-per-million", type=nonnegative_float,
        help="USD per million ordinary input tokens.",
    )
    pricing.add_argument(
        "--cached-input-price-per-million", type=nonnegative_float,
        help="USD per million cached input tokens.",
    )
    pricing.add_argument(
        "--cache-write-price-per-million", type=nonnegative_float,
        help="USD per million cache-write input tokens.",
    )
    pricing.add_argument(
        "--output-price-per-million", type=nonnegative_float,
        help="USD per million output tokens, including reasoning tokens.",
    )
    pricing.add_argument(
        "--max-cost-usd", type=nonnegative_float,
        help=(
            "Stop before starting another input once the estimated cost from "
            "recorded API token usage reaches this many USD. Requests already "
            "in flight can make the final estimate exceed the limit."
        ),
    )

    client = parser.add_argument_group("API client")
    client.add_argument("--api-key-env", help="Environment variable containing the provider API key.")
    client.add_argument("--base-url", help="Optional provider SDK base URL override.")
    client.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Concurrent requests. Start at one for gateways that enforce burst controls.",
    )
    client.add_argument("--request-chunk-size", type=int, default=64)
    client.add_argument("--timeout", type=float, default=600.0)
    client.add_argument("--max-retries", type=int, default=5)

    output = parser.add_argument_group("output/recovery")
    output.add_argument("--save-every", type=int, default=25)
    output.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--quiet", action="store_true")

    if configure_parser is not None:
        configure_parser(parser)

    args = parser.parse_args(argv)
    if args.start_index < 0 or (args.max_examples is not None and args.max_examples <= 0):
        parser.error("indices/counts must be positive (start-index may be zero)")
    if min(args.max_tokens, args.n, args.max_concurrency, args.request_chunk_size, args.save_every) <= 0:
        parser.error("generation, concurrency, chunk, and save values must be positive")
    if args.timeout <= 0 or args.max_retries < 0:
        parser.error("--timeout must be positive and --max-retries may not be negative")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.top_p is not None and not 0 <= args.top_p <= 1:
        parser.error("--top-p must be between 0 and 1")
    price_values = (
        args.input_price_per_million,
        args.cached_input_price_per_million,
        args.cache_write_price_per_million,
        args.output_price_per_million,
    )
    if any(value is not None for value in price_values) and not all(
        value is not None for value in price_values
    ):
        parser.error("all four cost-tracking prices must be specified together")
    if args.max_cost_usd is not None and not all(
        value is not None for value in price_values
    ):
        parser.error("--max-cost-usd requires all four cost-tracking prices")
    if not args.validate_only:
        if not args.model:
            parser.error("--model is required unless --validate-only is used")
        if args.output_dir is None:
            parser.error("--output-dir is required unless --validate-only is used")
    if args.overwrite:
        args.resume = False
    return args


def infer_provider(model: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    name = model.lower().split("/")[-1]
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Cannot infer an API provider from model {model!r}; pass --provider openai or --provider anthropic"
    )


def default_api_key_env(provider: str) -> str:
    if provider == "gateway":
        return "GATEWAY_KEY"
    return "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") in (None, "text", "input_text"):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def openai_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    allowed = {"system", "developer", "user", "assistant"}
    result = []
    for message in messages:
        role = str(message.get("role") or "")
        if role not in allowed:
            raise ValueError(f"OpenAI Responses API does not support prompt role {role!r}")
        result.append({"role": role, "content": _message_text(message.get("content"))})
    return result


def anthropic_messages(messages: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    system_parts: List[str] = []
    result: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        text = _message_text(message.get("content"))
        if role in ("system", "developer"):
            system_parts.append(text)
        elif role in ("user", "assistant"):
            result.append({"role": role, "content": text})
        else:
            raise ValueError(f"Anthropic Messages API does not support prompt role {role!r}")
    if not result:
        raise ValueError("Anthropic prompt has no user or assistant messages")
    return ("\n\n".join(system_parts) or None), result


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain_json(value: Any) -> Any:
    """Convert provider SDK values to JSON-compatible containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain_json(model_dump(mode="json"))
        except TypeError:
            return _plain_json(model_dump())
    return str(value)


def _source_metadata(value: Any) -> Optional[Dict[str, Any]]:
    source = {
        key: _get(value, key)
        for key in ("url", "title", "page_age")
        if _get(value, key) is not None
    }
    return source or None


def _citation_metadata(value: Any) -> Optional[Dict[str, Any]]:
    citation = {
        key: _get(value, key)
        for key in ("type", "url", "title", "cited_text", "start_index", "end_index")
        if _get(value, key) is not None
    }
    return citation or None


def _deduplicate_sources(values: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any]] = set()
    for value in values:
        key = (value.get("url"), value.get("title"))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _openai_web_search_metadata(response: Any) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    for item in _get(response, "output", []) or []:
        item_type = _get(item, "type")
        if item_type == "web_search_call":
            action = _plain_json(_get(item, "action"))
            calls.append({
                "id": _get(item, "id"),
                "status": _get(item, "status"),
                "action": action,
            })
            if isinstance(action, Mapping):
                for source in action.get("sources", []) or []:
                    metadata = _source_metadata(source)
                    if metadata:
                        sources.append(metadata)
        elif item_type == "message":
            for block in _get(item, "content", []) or []:
                for annotation in _get(block, "annotations", []) or []:
                    metadata = _citation_metadata(annotation)
                    if metadata:
                        citations.append(metadata)
                        source = _source_metadata(annotation)
                        if source:
                            sources.append(source)
    search_requests = sum(
        1
        for call in calls
        if isinstance(call.get("action"), Mapping)
        and call["action"].get("type") == "search"
    )
    return {
        "requests": search_requests,
        "tool_calls": len(calls),
        "calls": calls,
        "sources": _deduplicate_sources(sources),
        "citations": citations,
    }


def _anthropic_web_search_metadata(response: Any) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    for block in _get(response, "content", []) or []:
        block_type = _get(block, "type")
        if block_type == "server_tool_use" and _get(block, "name") == "web_search":
            calls.append({
                "id": _get(block, "id"),
                "name": "web_search",
                "input": _plain_json(_get(block, "input")),
            })
        elif block_type == "web_search_tool_result":
            content = _get(block, "content")
            if isinstance(content, (list, tuple)):
                for result in content:
                    if _get(result, "type") == "web_search_result":
                        metadata = _source_metadata(result)
                        if metadata:
                            sources.append(metadata)
        elif block_type == "text":
            for citation in _get(block, "citations", []) or []:
                metadata = _citation_metadata(citation)
                if metadata:
                    citations.append(metadata)
                    source = _source_metadata(citation)
                    if source:
                        sources.append(source)
    usage = _get(response, "usage")
    server_usage = _get(usage, "server_tool_use")
    raw_requests = _get(server_usage, "web_search_requests")
    return {
        "requests": int(raw_requests) if raw_requests is not None else len(calls),
        "tool_calls": len(calls),
        "calls": calls,
        "sources": _deduplicate_sources(sources),
        "citations": citations,
    }


def extract_openai_response(response: Any) -> Dict[str, Any]:
    """Normalize a Responses API object without depending on one SDK version."""
    summaries: List[str] = []
    for item in _get(response, "output", []) or []:
        if _get(item, "type") == "reasoning":
            summaries.extend(
                str(_get(part, "text")) for part in (_get(item, "summary", []) or [])
                if _get(part, "type") == "summary_text" and _get(part, "text")
            )
    text = _get(response, "output_text")
    if not isinstance(text, str):
        parts: List[str] = []
        for item in _get(response, "output", []) or []:
            if _get(item, "type") != "message":
                continue
            for block in _get(item, "content", []) or []:
                if _get(block, "type") == "output_text" and isinstance(_get(block, "text"), str):
                    parts.append(_get(block, "text"))
        text = "".join(parts)
    status = str(_get(response, "status") or "unknown")
    incomplete = _get(response, "incomplete_details")
    stop_reason = _get(incomplete, "reason") if incomplete is not None else None
    finish_reason = "length" if stop_reason in ("max_output_tokens", "max_tokens") else status
    usage = _get(response, "usage")
    input_details = _get(usage, "input_tokens_details")
    output_details = _get(usage, "output_tokens_details")
    reasoning_tokens = int(_get(output_details, "reasoning_tokens", 0) or 0)
    output_tokens = int(_get(usage, "output_tokens", 0) or 0)
    result = {
        "text": text,
        "reasoning_summary": "\n\n".join(summaries) or None,
        "reasoning_tokens": reasoning_tokens,
        "non_reasoning_output_tokens": output_tokens - reasoning_tokens,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "input_tokens": int(_get(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(_get(input_details, "cached_tokens", 0) or 0),
        "cache_write_input_tokens": int(
            _get(input_details, "cache_write_tokens", 0) or 0
        ),
        "input_token_details_available": input_details is not None,
        "output_tokens": output_tokens,
        "request_id": _get(response, "_request_id"),
        "response_id": _get(response, "id"),
    }
    web_search = _openai_web_search_metadata(response)
    if web_search["tool_calls"] or web_search["sources"] or web_search["citations"]:
        result["web_search"] = web_search
    return result


def extract_anthropic_response(response: Any) -> Dict[str, Any]:
    thinking_parts = [
        _get(block, "thinking") for block in (_get(response, "content", []) or [])
        if _get(block, "type") == "thinking" and isinstance(_get(block, "thinking"), str)
    ]
    parts = [
        _get(block, "text") for block in (_get(response, "content", []) or [])
        if _get(block, "type") == "text" and isinstance(_get(block, "text"), str)
    ]
    stop_reason = str(_get(response, "stop_reason") or "unknown")
    usage = _get(response, "usage")
    output_tokens = int(_get(usage, "output_tokens", 0) or 0)
    output_details = _get(usage, "output_tokens_details")
    raw_thinking_tokens = _get(output_details, "thinking_tokens")
    thinking_tokens = int(raw_thinking_tokens) if raw_thinking_tokens is not None else None
    stop_details = _plain_json(_get(response, "stop_details"))
    result = {
        "text": "".join(parts),
        "reasoning_summary": "\n\n".join(thinking_parts) or None,
        "reasoning_tokens": thinking_tokens,
        "non_reasoning_output_tokens": output_tokens - thinking_tokens if thinking_tokens is not None else None,
        "finish_reason": "length" if stop_reason == "max_tokens" else stop_reason,
        "stop_reason": stop_reason,
        "stop_details": stop_details,
        "input_tokens": int(_get(usage, "input_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "request_id": _get(response, "_request_id"),
        "response_id": _get(response, "id"),
    }
    web_search = _anthropic_web_search_metadata(response)
    if web_search["tool_calls"] or web_search["sources"] or web_search["citations"]:
        result["web_search"] = web_search
    return result


def extract_gateway_response(response: Any) -> Dict[str, Any]:
    choices = _get(response, "choices", []) or []
    choice = choices[0] if choices else None
    message = _get(choice, "message") if choice is not None else None
    finish_reason = str(_get(choice, "finish_reason") or "unknown")
    usage = _get(response, "usage")
    output_tokens = int(_get(usage, "completion_tokens", 0) or 0)
    output_details = _get(usage, "completion_tokens_details")
    reasoning_tokens = int(_get(output_details, "reasoning_tokens", 0) or 0)
    reasoning_content = _get(message, "reasoning_content") or _get(message, "reasoning")
    return {
        "text": _get(message, "content", "") or "",
        "reasoning_summary": str(reasoning_content) if reasoning_content else None,
        "reasoning_tokens": reasoning_tokens,
        "non_reasoning_output_tokens": output_tokens - reasoning_tokens,
        "finish_reason": "length" if finish_reason == "length" else finish_reason,
        "stop_reason": finish_reason,
        "input_tokens": int(_get(usage, "prompt_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "request_id": _get(response, "_request_id"),
        "response_id": _get(response, "id"),
    }


def _web_search_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "web_search", False))


def _web_search_domains(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    return (
        list(getattr(args, "web_search_allowed_domains", None) or []),
        list(getattr(args, "web_search_blocked_domains", None) or []),
    )


def openai_web_search_tool(args: argparse.Namespace) -> Dict[str, Any]:
    tool: Dict[str, Any] = {"type": "web_search"}
    context_size = getattr(args, "web_search_context_size", None)
    if context_size:
        tool["search_context_size"] = context_size
    allowed, blocked = _web_search_domains(args)
    if allowed or blocked:
        tool["filters"] = {
            "allowed_domains" if allowed else "blocked_domains": allowed or blocked,
        }
    return tool


def anthropic_web_search_tool(args: argparse.Namespace) -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": getattr(args, "anthropic_web_search_version", "web_search_20250305"),
        "name": "web_search",
        "max_uses": int(getattr(args, "web_search_max_uses", 5)),
    }
    allowed, blocked = _web_search_domains(args)
    if allowed:
        tool["allowed_domains"] = allowed
    elif blocked:
        tool["blocked_domains"] = blocked
    return tool


def _merge_web_search_metadata(results: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    searches = [result.get("web_search") for result in results if result.get("web_search")]
    if not searches:
        return None
    sources = [source for search in searches for source in search.get("sources", [])]
    return {
        "requests": sum(int(search.get("requests") or 0) for search in searches),
        "tool_calls": sum(int(search.get("tool_calls") or 0) for search in searches),
        "calls": [call for search in searches for call in search.get("calls", [])],
        "sources": _deduplicate_sources(sources),
        "citations": [
            citation for search in searches for citation in search.get("citations", [])
        ],
    }


async def _anthropic_request_with_continuations(
    send: Any,
    payload: Mapping[str, Any],
    max_continuations: int,
) -> Dict[str, Any]:
    """Run a Messages request and resume server-tool ``pause_turn`` responses."""
    current = dict(payload)
    current["messages"] = list(payload["messages"])
    extracted: List[Dict[str, Any]] = []
    for continuation in range(max_continuations + 1):
        response = await send(current)
        extracted.append(extract_anthropic_response(response))
        if _get(response, "stop_reason") != "pause_turn":
            break
        if continuation == max_continuations:
            raise RuntimeError(
                f"Anthropic web search exceeded {max_continuations} pause_turn continuations"
            )
        current = dict(current)
        current["messages"] = list(current["messages"]) + [{
            "role": "assistant",
            "content": _plain_json(_get(response, "content", []) or []),
        }]
    final = dict(extracted[-1])
    final["text"] = "".join(str(item.get("text") or "") for item in extracted)
    summaries = [str(item["reasoning_summary"]) for item in extracted if item.get("reasoning_summary")]
    final["reasoning_summary"] = "\n\n".join(summaries) or None
    final["input_tokens"] = sum(int(item.get("input_tokens") or 0) for item in extracted)
    final["output_tokens"] = sum(int(item.get("output_tokens") or 0) for item in extracted)
    reasoning_values = [item.get("reasoning_tokens") for item in extracted]
    if all(value is not None for value in reasoning_values):
        final["reasoning_tokens"] = sum(int(value) for value in reasoning_values)
        final["non_reasoning_output_tokens"] = (
            final["output_tokens"] - final["reasoning_tokens"]
        )
    else:
        final["reasoning_tokens"] = None
        final["non_reasoning_output_tokens"] = None
    web_search = _merge_web_search_metadata(extracted)
    if web_search:
        final["web_search"] = web_search
    return final


async def request_openai(client: Any, args: argparse.Namespace, messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasoning: Dict[str, Any] = {"summary": "detailed"}
    if args.reasoning_effort:
        reasoning["effort"] = args.reasoning_effort
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "input": openai_messages(messages),
        "max_output_tokens": args.max_tokens,
        "reasoning": reasoning,
        "store": False,
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if _web_search_enabled(args):
        kwargs["tools"] = [openai_web_search_tool(args)]
        kwargs["tool_choice"] = "auto"
        kwargs["include"] = ["web_search_call.action.sources"]
    return extract_openai_response(await client.responses.create(**kwargs))


async def request_anthropic(client: Any, args: argparse.Namespace, messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    system, api_messages = anthropic_messages(messages)
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "messages": api_messages,
        "max_tokens": args.max_tokens,
        "thinking": {"type": "adaptive", "display": "summarized"},
    }
    if system is not None:
        kwargs["system"] = system
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.reasoning_effort:
        kwargs["output_config"] = {"effort": args.reasoning_effort}
    if not _web_search_enabled(args):
        return extract_anthropic_response(await client.messages.create(**kwargs))
    kwargs["tools"] = [anthropic_web_search_tool(args)]

    async def send(payload: Mapping[str, Any]) -> Any:
        return await client.messages.create(**payload)

    return await _anthropic_request_with_continuations(
        send,
        kwargs,
        int(getattr(args, "web_search_max_continuations", 8)),
    )


async def request_gateway(client: Any, args: argparse.Namespace, messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if str(args.model).startswith("openai/"):
        reasoning: Dict[str, Any] = {"summary": "detailed"}
        if args.reasoning_effort:
            reasoning["effort"] = args.reasoning_effort
        payload = {
            "model": str(args.model).split("/", 1)[1],
            "input": openai_messages(messages),
            "reasoning": reasoning,
            "max_output_tokens": args.max_tokens,
            "store": False,
        }
        if _web_search_enabled(args):
            payload["tools"] = [openai_web_search_tool(args)]
            payload["tool_choice"] = "auto"
            payload["include"] = ["web_search_call.action.sources"]
        response = await _gateway_post(client, "openai/v1/responses", payload, args.max_retries)
        return extract_openai_response(response)

    if str(args.model).startswith("anthropic/"):
        system, api_messages = anthropic_messages(messages)
        payload: Dict[str, Any] = {
            "model": str(args.model).split("/", 1)[1],
            "messages": api_messages,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if system is not None:
            payload["system"] = system
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if args.reasoning_effort:
            payload["output_config"] = {"effort": args.reasoning_effort}
        if not _web_search_enabled(args):
            response = await _gateway_post(
                client,
                "anthropic/v1/messages",
                payload,
                args.max_retries,
                headers={"anthropic-version": "2023-06-01"},
            )
            return extract_anthropic_response(response)
        payload["tools"] = [anthropic_web_search_tool(args)]

        async def send(current: Mapping[str, Any]) -> Any:
            return await _gateway_post(
                client,
                "anthropic/v1/messages",
                current,
                args.max_retries,
                headers={"anthropic-version": "2023-06-01"},
            )

        return await _anthropic_request_with_continuations(
            send,
            payload,
            int(getattr(args, "web_search_max_continuations", 8)),
        )

    if _web_search_enabled(args):
        raise ValueError(
            "gateway web search requires an openai/... or anthropic/... model prefix"
        )
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": openai_messages(messages),
        "max_completion_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    response = await _gateway_post(client, "compat/chat/completions", payload, args.max_retries)
    return extract_gateway_response(response)


async def _gateway_post(
    client: Any,
    path: str,
    payload: Mapping[str, Any],
    max_retries: int,
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install the gateway HTTP client: python -m pip install httpx") from exc
    response: Optional[httpx.Response] = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(path, json=payload, headers=headers)
        except httpx.RequestError:
            if attempt == max_retries:
                raise
        else:
            transient = response.status_code in (408, 409, 429) or response.status_code >= 500
            if not transient or attempt == max_retries:
                break
        await asyncio.sleep(min(2 ** attempt, 30))
    assert response is not None
    request_id = response.headers.get("x-request-id") or response.headers.get("x-ai-gateway-request-id")
    if response.is_error:
        body = response.text.strip().replace("\n", " ")[:2000]
        raise RuntimeError(f"Gateway HTTP {response.status_code} (request_id={request_id}): {body}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Gateway returned non-JSON content (request_id={request_id})") from exc
    if isinstance(payload, dict):
        payload["_request_id"] = request_id
    return payload


def create_client(provider: str, args: argparse.Namespace) -> Any:
    key_env = args.api_key_env or default_api_key_env(provider)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"Set {key_env} to run {provider} API evaluation")
    if provider == "gateway":
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install the gateway HTTP client: python -m pip install httpx") from exc
        base_url = (args.base_url or "https://gateway.engineering.jhu.edu/gateway").rstrip("/") + "/"
        return httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "matsci-evaluator/1.0",
            },
            timeout=args.timeout,
        )
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK: python -m pip install openai") from exc
        return AsyncOpenAI(**kwargs)
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError("Install the Anthropic SDK: python -m pip install anthropic") from exc
    return AsyncAnthropic(**kwargs)


def pricing_from_args(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    input_price = getattr(args, "input_price_per_million", None)
    if input_price is None:
        return None
    return {
        "input": float(input_price),
        "cached_input": float(args.cached_input_price_per_million),
        "cache_write": float(args.cache_write_price_per_million),
        "output": float(args.output_price_per_million),
    }


def estimate_cost(
    usage: Mapping[str, Any],
    pricing: Optional[Mapping[str, float]],
    completed_examples: int,
) -> Optional[Dict[str, Any]]:
    """Estimate token cost from recorded usage and an explicit price snapshot."""
    if pricing is None:
        return None
    ordinary = int(usage.get("uncached_input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    cache_write = int(usage.get("cache_write_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    candidates = int(usage.get("candidate_generations") or 0)
    ordinary_cost = ordinary * float(pricing["input"]) / TOKENS_PER_MILLION
    cached_cost = cached * float(pricing["cached_input"]) / TOKENS_PER_MILLION
    cache_write_cost = (
        cache_write * float(pricing["cache_write"]) / TOKENS_PER_MILLION
    )
    output_cost = output * float(pricing["output"]) / TOKENS_PER_MILLION
    input_cost = ordinary_cost + cached_cost + cache_write_cost
    total = input_cost + output_cost
    result = {
        "currency": "USD",
        "estimated": True,
        "pricing_per_million_tokens": dict(pricing),
        "ordinary_input_cost_usd": ordinary_cost,
        "cached_input_cost_usd": cached_cost,
        "cache_write_cost_usd": cache_write_cost,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total,
        "average_cost_per_completed_input_usd": (
            total / completed_examples if completed_examples else None
        ),
        "average_cost_per_generation_usd": total / candidates if candidates else None,
        "input_token_details_missing_generations": int(
            usage.get("input_token_details_missing_generations") or 0
        ),
        "maximum_input_tokens_per_generation": int(
            usage.get("maximum_input_tokens_per_generation") or 0
        ),
        "long_context_pricing_applied": False,
    }
    if result["maximum_input_tokens_per_generation"] > 272_000:
        result["warning"] = (
            "standard token rates were used although at least one prompt exceeded "
            "272,000 input tokens; apply the model's long-context pricing manually"
        )
    return result


def make_run_config(args: argparse.Namespace, provider: str, profile: Mapping[str, Any]) -> Dict[str, Any]:
    config = {
        "format_version": 1,
        "input": dict(profile),
        "provider": provider,
        "api": {
            "openai": "responses",
            "anthropic": "messages",
            "gateway": "native OpenAI Responses / native Anthropic Messages / compat Chat Completions",
        }[provider],
        "model": args.model,
        "api_key_env": args.api_key_env or default_api_key_env(provider),
        "base_url": args.base_url,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_summary": (
                "detailed" if provider == "openai" or str(args.model).startswith("openai/")
                else "summarized" if provider == "anthropic" or str(args.model).startswith("anthropic/")
                else "provider_default"
            ),
            "n": args.n,
        },
        "client": {
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "max_concurrency": args.max_concurrency,
            "request_chunk_size": args.request_chunk_size,
        },
    }
    pricing = pricing_from_args(args)
    if pricing is not None:
        config["cost_tracking"] = {
            "currency": "USD",
            "pricing_per_million_tokens": pricing,
            "source": "explicit evaluator arguments",
            "long_context_pricing_applied": False,
        }
    if _web_search_enabled(args):
        allowed, blocked = _web_search_domains(args)
        config["web_search"] = {
            "enabled": True,
            "tool_choice": "auto",
            "max_uses": getattr(args, "web_search_max_uses", None),
            "max_pause_turn_continuations": getattr(
                args, "web_search_max_continuations", None
            ),
            "openai_tool_type": "web_search",
            "openai_search_context_size": getattr(
                args, "web_search_context_size", None
            ),
            "anthropic_tool_type": getattr(
                args, "anthropic_web_search_version", None
            ),
            "allowed_domains": allowed,
            "blocked_domains": blocked,
        }
    return config


def comparable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    input_config = config.get("input", {})
    return {
        "input_sha256": input_config.get("sha256"),
        "input_source_sha256": input_config.get(
            "source_sha256", input_config.get("sha256")
        ),
        "selected_start_index": input_config.get("selected_start_index"),
        "selected_rows": input_config.get("selected_rows"),
        **{
            key: config.get(key)
            for key in (
                "provider", "api", "model", "base_url", "sampling", "web_search",
                "cost_tracking",
            )
        },
    }


def resume_config_compatible(
    existing_config: Mapping[str, Any],
    requested_config: Mapping[str, Any],
) -> bool:
    """Allow an existing run to grow while rejecting other config changes."""
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

    # CIF fingerprints cover all selected CIFs, so they change when the prefix
    # grows. The stable source Parquet fingerprint and all API settings must match.
    existing.pop("input_sha256", None)
    requested.pop("input_sha256", None)
    return existing == requested


def prepare_outputs(args: argparse.Namespace, config: Mapping[str, Any]) -> Tuple[Path, Path]:
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
            raise ValueError("cannot resume: run_config.json is missing")
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not resume_config_compatible(old_config, config):
            raise ValueError("existing predictions use an incompatible dataset/model/config")
        if comparable_config(old_config) != comparable_config(config):
            previous_rows = old_config.get("input", {}).get("selected_rows")
            requested_rows = config.get("input", {}).get("selected_rows")
            atomic_write_json(config_path, config)
            print(
                f"Extending saved run selection from {previous_rows} to "
                f"{requested_rows} rows: {args.output_dir}"
            )
    else:
        atomic_write_json(config_path, config)
    return predictions, metrics_path


def score_prediction_records(
    rows: Sequence[Mapping[str, Any]], records: Mapping[int, Mapping[str, Any]],
) -> Tuple[list, Dict[str, Any]]:
    scored = []
    input_tokens = cached_input_tokens = cache_write_input_tokens = 0
    output_tokens = truncations = web_search_requests = 0
    candidate_generations = input_detail_missing = max_input_tokens = 0
    finish_reasons: Counter[str] = Counter()
    for row in rows:
        index = int(row["row_index"])
        record = records.get(index)
        if record is None:
            raise ValueError(f"missing prediction for row_index {index}")
        if str(record.get("id")) != str(row["id"]):
            raise ValueError(f"id mismatch for row_index {index}")
        candidates = record.get("candidates") or []
        if not candidates:
            raise ValueError(f"no candidates for row_index {index}")
        scores = []
        for rank, candidate in enumerate(candidates):
            scores.append(score_response(str(candidate.get("text") or ""), row))
            candidate_input_tokens = int(candidate.get("input_tokens") or 0)
            candidate_cached_tokens = int(candidate.get("cached_input_tokens") or 0)
            candidate_cache_write_tokens = int(
                candidate.get("cache_write_input_tokens") or 0
            )
            if candidate_cached_tokens + candidate_cache_write_tokens > candidate_input_tokens:
                raise ValueError(
                    f"row_index {index} candidate {rank}: cached/cache-write input "
                    "tokens exceed total input tokens"
                )
            input_tokens += candidate_input_tokens
            cached_input_tokens += candidate_cached_tokens
            cache_write_input_tokens += candidate_cache_write_tokens
            output_tokens += int(candidate.get("output_tokens") or 0)
            candidate_generations += 1
            max_input_tokens = max(max_input_tokens, candidate_input_tokens)
            input_detail_missing += int(
                not bool(candidate.get("input_token_details_available", False))
            )
            web_search_requests += int(
                (candidate.get("web_search") or {}).get("requests") or 0
            )
            reason = str(candidate.get("finish_reason") or "unknown")
            finish_reasons[reason] += 1
            truncations += int(rank == 0 and reason == "length")
        scored.append((row, scores[0], scores))
    usage = {
        "input_tokens": input_tokens,
        "prompt_tokens": input_tokens,
        "uncached_input_tokens": (
            input_tokens - cached_input_tokens - cache_write_input_tokens
        ),
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "input_token_details_missing_generations": input_detail_missing,
        "maximum_input_tokens_per_generation": max_input_tokens,
        "output_tokens": output_tokens,
        "candidate_generations": candidate_generations,
        "web_search_requests": web_search_requests,
        "finish_reason_counts_all_candidates": dict(finish_reasons),
        "top1_length_truncation_count": truncations,
        "top1_length_truncation_rate": truncations / len(rows) if rows else None,
    }
    return scored, usage


async def generate_row(
    provider: str,
    client: Any,
    args: argparse.Namespace,
    row: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    request = {
        "openai": request_openai,
        "anthropic": request_anthropic,
        "gateway": request_gateway,
    }[provider]
    prompt_messages = row["prompt_messages"]
    prompt_transform = getattr(args, "prompt_transform", None)
    if prompt_transform is not None:
        prompt_messages = prompt_transform(row)

    async def one(rank: int) -> Dict[str, Any]:
        async with semaphore:
            result = await request(client, args, prompt_messages)
        if result.get("finish_reason") == "refusal":
            raise ProviderRefusalError(row, result.get("stop_details"))
        score = score_response(result["text"], row)
        candidate = {
            "rank": rank,
            "text": result["text"],
            "reasoning_summary": result.get("reasoning_summary"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "non_reasoning_output_tokens": result.get("non_reasoning_output_tokens"),
            "parsed_formula": score["predicted_formula"],
            "parsed_geometry": score["parsed_geometry"],
            "finish_reason": result["finish_reason"],
            "stop_reason": result["stop_reason"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "request_id": result["request_id"],
            "response_id": result["response_id"],
            "scores": score,
        }
        if "cached_input_tokens" in result:
            candidate["cached_input_tokens"] = result["cached_input_tokens"]
            candidate["cache_write_input_tokens"] = result[
                "cache_write_input_tokens"
            ]
            candidate["input_token_details_available"] = result[
                "input_token_details_available"
            ]
        if result.get("stop_details") is not None:
            candidate["stop_details"] = result["stop_details"]
        if result.get("web_search") is not None:
            candidate["web_search"] = result["web_search"]
        elif _web_search_enabled(args):
            candidate["web_search"] = {
                "requests": 0,
                "tool_calls": 0,
                "calls": [],
                "sources": [],
                "citations": [],
            }
        return candidate

    candidates = await asyncio.gather(*(one(rank) for rank in range(args.n)))
    record = {
        "row_index": int(row["row_index"]),
        "id": row["id"],
        "material_id": row["metadata"].get("material_id"),
        "provider": provider,
        "model": args.model,
        "prompt_tokens": sum(candidate["input_tokens"] for candidate in candidates),
        "candidates": candidates,
    }
    if row.get("rag_document_keys"):
        record["rag_document_keys"] = list(row["rag_document_keys"])
    if row.get("cif_context"):
        record["cif_context"] = dict(row["cif_context"])
    if row.get("distilled_reasoning_context"):
        record["distilled_reasoning_context"] = dict(
            row["distilled_reasoning_context"]
        )
    return record


async def generate_pending(
    provider: str,
    client: Any,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    existing: Mapping[int, Mapping[str, Any]],
    path: Path,
    initial_scored: list,
    checkpoint: Any,
    resumed_usage: Mapping[str, Any],
    pricing: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    pending = [row for row in rows if int(row["row_index"]) not in existing]
    semaphore = asyncio.Semaphore(args.max_concurrency)
    started = time.perf_counter()
    generated = failed = input_tokens = output_tokens = 0
    cached_input_tokens = cache_write_input_tokens = 0
    candidate_generations = input_detail_missing = max_input_tokens = 0
    budget_exhausted = False
    failed_request_details: List[Dict[str, Any]] = []
    progress = list(initial_scored)
    exact_matches = sum(bool(score.get("exact_formula")) for _, score, _ in progress)
    system_matches = sum(bool(score.get("chemical_system_match")) for _, score, _ in progress)
    total_output_tokens = sum(
        int(candidate.get("output_tokens") or 0)
        for record in existing.values()
        for candidate in (record.get("candidates") or [])
    )

    def runtime() -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {
            "resumed_examples": len(existing),
            "generated_examples_this_run": generated,
            "failed_examples_this_run": failed,
            "generation_wall_seconds_this_run": elapsed,
            "input_tokens_this_run": input_tokens,
            "prompt_tokens_this_run": input_tokens,
            "cached_input_tokens_this_run": cached_input_tokens,
            "cache_write_input_tokens_this_run": cache_write_input_tokens,
            "uncached_input_tokens_this_run": (
                input_tokens - cached_input_tokens - cache_write_input_tokens
            ),
            "input_token_details_missing_generations_this_run": input_detail_missing,
            "maximum_input_tokens_per_generation_this_run": max_input_tokens,
            "output_tokens_this_run": output_tokens,
            "candidate_generations_this_run": candidate_generations,
            "failed_request_details_this_run": list(failed_request_details),
            "examples_per_second_this_run": generated / elapsed if elapsed else None,
            "tokens_per_second_this_run": (input_tokens + output_tokens) / elapsed if elapsed else None,
            "max_cost_usd": args.max_cost_usd,
            "budget_exhausted": budget_exhausted,
            "budget_enforcement_granularity": (
                "input" if args.max_cost_usd is not None else None
            ),
        }

    def recorded_cost_usd() -> Optional[float]:
        cost = estimate_cost(
            current_usage(resumed_usage, runtime()),
            pricing,
            len(existing) + generated,
        )
        return None if cost is None else float(cost["total_cost_usd"])

    def progress_details() -> str:
        completed = len(existing) + generated
        exact_rate = 100 * exact_matches / completed if completed else 0.0
        system_rate = 100 * system_matches / completed if completed else 0.0
        details = (
            f"exact {exact_rate:.1f}% | system {system_rate:.1f}% | "
            f"output {total_output_tokens:,} tok"
        )
        cost = recorded_cost_usd()
        if cost is not None:
            details += f" | cost ${cost:.6f}"
            if args.max_cost_usd is not None:
                details += f" / ${args.max_cost_usd:.6f} max"
        return details

    tracker = ProgressTracker(
        len(rows), len(existing), enabled=not args.quiet, details=progress_details(),
    )

    async def generate_with_row(row: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Any]:
        try:
            result = await generate_row(provider, client, args, row, semaphore)
        except BaseException as exc:
            result = exc
        return row, result

    mode = "a" if path.exists() else "w"
    try:
        with path.open(mode, encoding="utf-8") as handle:
            begin = 0
            while begin < len(pending):
                current_cost = recorded_cost_usd()
                if (
                    args.max_cost_usd is not None
                    and current_cost is not None
                    and current_cost >= args.max_cost_usd
                ):
                    budget_exhausted = True
                    break
                completed = len(existing) + generated
                until_save = args.save_every - completed % args.save_every
                # With a cost ceiling, launch only one input at a time so any
                # overshoot is bounded to that input's candidate generations.
                size = 1 if args.max_cost_usd is not None else min(
                    args.request_chunk_size, until_save, len(pending) - begin
                )
                batch = pending[begin:begin + size]
                begin += size
                first_error: Optional[BaseException] = None
                tasks = [asyncio.create_task(generate_with_row(row)) for row in batch]
                for task in asyncio.as_completed(tasks):
                    row, result = await task
                    if isinstance(result, BaseException):
                        first_error = first_error or result
                        failed += 1
                        if isinstance(result, ProviderRefusalError):
                            failed_request_details.append(result.as_record())
                        else:
                            failed_request_details.append({
                                "row_index": int(row["row_index"]),
                                "id": str(row["id"]),
                                "error_type": type(result).__name__,
                                "message": str(result),
                            })
                        checkpoint(progress, runtime())
                        tracker.advance(success=False, details=progress_details())
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    generated += 1
                    scores = [candidate["scores"] for candidate in result["candidates"]]
                    progress.append((row, scores[0], scores))
                    exact_matches += int(bool(scores[0].get("exact_formula")))
                    system_matches += int(bool(scores[0].get("chemical_system_match")))
                    result_input_tokens = sum(
                        int(candidate["input_tokens"])
                        for candidate in result["candidates"]
                    )
                    input_tokens += result_input_tokens
                    cached_input_tokens += sum(
                        int(candidate.get("cached_input_tokens") or 0)
                        for candidate in result["candidates"]
                    )
                    cache_write_input_tokens += sum(
                        int(candidate.get("cache_write_input_tokens") or 0)
                        for candidate in result["candidates"]
                    )
                    candidate_generations += len(result["candidates"])
                    input_detail_missing += sum(
                        not bool(candidate.get("input_token_details_available", False))
                        for candidate in result["candidates"]
                    )
                    max_input_tokens = max(
                        max_input_tokens,
                        *(int(candidate["input_tokens"]) for candidate in result["candidates"]),
                    )
                    result_output_tokens = sum(
                        int(candidate["output_tokens"]) for candidate in result["candidates"]
                    )
                    output_tokens += result_output_tokens
                    total_output_tokens += result_output_tokens
                    checkpoint(progress, runtime())
                    tracker.advance(details=progress_details())
                if first_error is not None:
                    raise first_error
    finally:
        tracker.close()
    return runtime()


def current_usage(
    resumed_usage: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Dict[str, Any]:
    """Combine resumed and current-invocation counters for checkpoint metrics."""
    final = runtime.get("final_record_usage")
    if isinstance(final, Mapping):
        return dict(final)
    additive = {
        "input_tokens": "input_tokens_this_run",
        "prompt_tokens": "prompt_tokens_this_run",
        "uncached_input_tokens": "uncached_input_tokens_this_run",
        "cached_input_tokens": "cached_input_tokens_this_run",
        "cache_write_input_tokens": "cache_write_input_tokens_this_run",
        "input_token_details_missing_generations": (
            "input_token_details_missing_generations_this_run"
        ),
        "output_tokens": "output_tokens_this_run",
        "candidate_generations": "candidate_generations_this_run",
    }
    usage = {
        output_key: int(resumed_usage.get(output_key) or 0)
        + int(runtime.get(runtime_key) or 0)
        for output_key, runtime_key in additive.items()
    }
    usage["maximum_input_tokens_per_generation"] = max(
        int(resumed_usage.get("maximum_input_tokens_per_generation") or 0),
        int(runtime.get("maximum_input_tokens_per_generation_this_run") or 0),
    )
    usage["web_search_requests"] = int(
        resumed_usage.get("web_search_requests") or 0
    )
    return usage


async def async_main(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], profile: Mapping[str, Any]) -> None:
    provider = infer_provider(str(args.model), args.provider)
    if provider == "gateway" and "/" not in str(args.model):
        raise ValueError("gateway models must be provider-prefixed, for example openai/gpt-5.2")
    config = make_run_config(args, provider, profile)
    predictions, metrics_path = prepare_outputs(args, config)
    existing = read_prediction_records(predictions)
    selected = {int(row["row_index"]) for row in rows}
    extra = set(existing) - selected
    if extra:
        raise ValueError(f"prediction file has rows outside this selection: {sorted(extra)[:5]}")
    completed_rows = [row for row in rows if int(row["row_index"]) in existing]
    initial_scored, resumed_usage = score_prediction_records(completed_rows, existing) if completed_rows else ([], {})

    pricing = pricing_from_args(args)

    def checkpoint(scored: Sequence[Any], runtime: Mapping[str, Any]) -> None:
        usage = current_usage(resumed_usage, runtime)
        if len(scored) == len(rows):
            status = "complete"
        elif runtime.get("budget_exhausted"):
            status = "budget_exhausted"
        else:
            status = "in_progress"
        payload = {
            "status": status,
            "completed_examples": len(scored),
            "selected_examples": len(rows),
            "dataset_profile": profile["validation"],
            "metrics": summarize_scored(scored),
            "usage": usage,
            "usage_from_resumed_records": resumed_usage,
            "runtime_this_invocation": dict(runtime),
            "updated_unix_time": time.time(),
        }
        cost = estimate_cost(usage, pricing, len(scored))
        if cost is not None:
            payload["estimated_cost"] = cost
        atomic_write_json(metrics_path, payload)

    if len(existing) == len(rows):
        checkpoint(initial_scored, {"resumed_examples": len(existing), "generated_examples_this_run": 0})
        completed_cost = estimate_cost(resumed_usage, pricing, len(existing))
        cost_text = (
            f" Estimated cost: ${completed_cost['total_cost_usd']:.6f}."
            if completed_cost is not None else ""
        )
        print(
            f"All {len(rows)} examples already complete.{cost_text} "
            f"Metrics: {metrics_path}"
        )
        return

    starting_cost = estimate_cost(resumed_usage, pricing, len(existing))
    if (
        args.max_cost_usd is not None
        and starting_cost is not None
        and starting_cost["total_cost_usd"] >= args.max_cost_usd
    ):
        runtime = {
            "resumed_examples": len(existing),
            "generated_examples_this_run": 0,
            "max_cost_usd": args.max_cost_usd,
            "budget_exhausted": True,
            "budget_enforcement_granularity": "input",
        }
        checkpoint(initial_scored, runtime)
        print(
            f"Cost limit reached before new requests: estimated cost "
            f"${starting_cost['total_cost_usd']:.6f} >= "
            f"${args.max_cost_usd:.6f}; stopped at {len(existing)}/{len(rows)} "
            f"inputs. Metrics: {metrics_path}"
        )
        return

    client = create_client(provider, args)
    try:
        runtime = await generate_pending(
            provider, client, args, rows, existing, predictions, initial_scored, checkpoint,
            resumed_usage, pricing,
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
    final_records = read_prediction_records(predictions)
    final_completed_rows = [
        row for row in rows if int(row["row_index"]) in final_records
    ]
    final_scored, usage = score_prediction_records(final_completed_rows, final_records)
    runtime["final_record_usage"] = usage
    checkpoint(final_scored, runtime)
    final_cost = estimate_cost(usage, pricing, len(final_scored))
    cost_text = (
        f" Estimated cost: ${final_cost['total_cost_usd']:.6f}."
        if final_cost is not None else ""
    )
    if runtime.get("budget_exhausted"):
        print(
            f"Cost limit reached; stopped at {len(final_scored)}/{len(rows)} inputs."
            f"{cost_text} Limit: ${args.max_cost_usd:.6f}. Metrics: {metrics_path}"
        )
    else:
        print(f"Completed {len(rows)} examples.{cost_text} Metrics: {metrics_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows, profile = load_rows(args.input_parquet, args.start_index, args.max_examples)
    profile["validation"] = validate_rows(rows)
    if args.validate_only:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return
    asyncio.run(async_main(args, rows, profile))


if __name__ == "__main__":
    main()
