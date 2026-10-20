#!/usr/bin/env python3
"""
Supervised fine-tuning with TRL, shared by the materials-science and
regulatory-genomics stage-1 and stage-2 launchers.

Expected Parquet schema (other columns are ignored):
  messages: list[{"role": "user"|"assistant", "content": "..."}]

This script:
  - Loads train/eval parquet files.
  - Removes any system messages if present.
  - Converts messages into TRL conversational prompt/completion format:
      {
        "prompt": [{"role": "user", "content": "..."}],
        "completion": [{"role": "assistant", "content": "..."}]
      }
  - Uses completion_only_loss=True, so loss is computed only on assistant output.
  - Supports full fine-tuning, LoRA, and QLoRA, including parameter-level LoRA
    on GPT-OSS mixture-of-experts weights.

Example:
  python train/train.py \\
    --model_name_or_path openai/gpt-oss-20b \\
    --train_file data/materials_science/data/stage1/ratio_0p500/variant_000/train.parquet \\
    --output_dir outputs/example_lora \\
    --use_lora \\
    --bf16
"""

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


SIMPLE_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
User: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}{{ eos_token }}

{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}"""

GPT_OSS_MOE_LORA_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.router.weight",
)


def should_force_trust_remote_code(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path)
    if not path.exists() or not path.is_dir():
        return False

    if (path / "configuration_openlm.py").exists() or (path / "modeling_openlm.py").exists():
        return True

    config_path = path / "config.json"
    if not config_path.exists():
        return False

    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return False

    return "auto_map" in config


def infer_tokenizer_source(model_name_or_path: str, explicit_tokenizer_name_or_path: Optional[str]) -> str:
    if explicit_tokenizer_name_or_path:
        return explicit_tokenizer_name_or_path

    path = Path(model_name_or_path)
    if not path.exists() or not path.is_dir():
        return model_name_or_path

    tokenizer_artifacts = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    if any((path / name).exists() for name in tokenizer_artifacts):
        return model_name_or_path

    config_yaml_path = path / "config.yaml"
    if config_yaml_path.exists():
        try:
            config_yaml = yaml.safe_load(config_yaml_path.read_text())
            tokenizer_name_or_path = (config_yaml.get("tokenizer") or {}).get("tokenizer_name_or_path")
            if tokenizer_name_or_path:
                print(
                    "No tokenizer artifacts found in model directory; "
                    f"falling back to tokenizer_name_or_path={tokenizer_name_or_path!r} from config.yaml."
                )
                return tokenizer_name_or_path
        except Exception:
            pass

    return model_name_or_path


def should_force_safe_attn_implementation(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path)
    if not path.exists() or not path.is_dir():
        return False
    return (path / "modeling_openlm.py").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL SFT training script for chat-formatted Parquet data.")

    # Data
    parser.add_argument("--train_file", type=Path, required=True, help="Training parquet file.")
    parser.add_argument("--eval_file", type=Path, default=None, help="Optional validation parquet file.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer source override. If omitted and the model directory has no tokenizer files, "
            "the script will try to recover tokenizer_name_or_path from config.yaml."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="Optional attention implementation for transformers.from_pretrained.",
    )

    # Chat template
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help=(
            "Optional TRL chat_template_path. Useful if the model tokenizer has no chat template. "
            "Example: HuggingFaceTB/SmolLM3-3B"
        ),
    )
    parser.add_argument(
        "--use_simple_chat_template_if_missing",
        action="store_true",
        help=(
            "If tokenizer has no chat_template and --chat_template_path is not set, "
            "install a minimal User:/Assistant: chat template."
        ),
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="Optional EOS token override for SFTConfig, e.g. <|im_end|> for some Qwen chat templates.",
    )

    # Precision / quantization
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_qlora", action="store_true", help="Load model in 4-bit and train LoRA.")
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto-if-qlora",
        help=(
            "Device map for from_pretrained. Use 'auto-if-qlora' to use auto only for single-GPU QLoRA. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
    )
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Train with LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help=(
            "Comma-separated target modules or 'all-linear'. "
            "For Llama/Qwen, common explicit value is "
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
    parser.add_argument(
        "--lora_target_parameters",
        type=str,
        default="auto",
        help=(
            "Comma-separated raw parameter suffixes to receive LoRA adapters, 'none' to disable, "
            "or 'auto' to include fused GPT-OSS MoE expert projections and router weights when "
            "model_type is gpt_oss. Parameter-level LoRA requires PEFT >= 0.17.1."
        ),
    )

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        type=json.loads,
        default=None,
        help='Optional JSON object passed to the LR scheduler, e.g. \'{"min_lr": 1e-5}\'.',
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Distributed sharding
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable Fully Sharded Data Parallel training. Launch with torchrun/torch.distributed.run.",
    )
    parser.add_argument(
        "--fsdp_config",
        type=Path,
        default=None,
        help="Optional Transformers FSDP JSON configuration file.",
    )

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--packing", action="store_true", help="Enable TRL packing.")
    parser.add_argument("--optim", type=str, default="adamw_torch")

    # Logging / saving / eval
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Comma-separated integrations: none, wandb, tensorboard, etc.",
    )
    parser.add_argument("--run_name", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument(
        "--debug_print_examples",
        type=int,
        default=2,
        help="Print N converted prompt/completion examples before training.",
    )

    return parser.parse_args()


def get_torch_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 or --fp16, not both.")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def resolve_device_map(args: argparse.Namespace) -> Optional[str]:
    if args.device_map == "none":
        return None

    if args.device_map == "auto-if-qlora":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if args.use_qlora and world_size == 1:
            return "auto"
        return None

    return args.device_map


def parse_lora_target_modules(value: str):
    value = value.strip()
    if value == "all-linear":
        return "all-linear"
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_lora_target_parameters(value: str, model_type: str) -> List[str]:
    normalized = value.strip()
    if normalized.lower() == "auto":
        if model_type == "gpt_oss":
            return list(GPT_OSS_MOE_LORA_TARGET_PARAMETERS)
        return []
    if normalized.lower() in {"", "none"}:
        return []
    return [item.strip() for item in normalized.split(",") if item.strip()]


def validate_lora_target_parameters(model, targets: List[str]) -> None:
    parameter_names = [name for name, _ in model.named_parameters()]
    missing = []
    for target in targets:
        matches = [
            name
            for name in parameter_names
            if name == target or name.endswith("." + target)
        ]
        if not matches:
            missing.append(target)
            continue
        print(f"LoRA raw-parameter target {target!r}: matched {len(matches)} tensors")

    if missing:
        raise ValueError(
            "LoRA raw-parameter targets were not found in the loaded model: "
            + ", ".join(missing)
        )


def mxfp4_dequantization_config():
    try:
        from transformers import Mxfp4Config
    except ImportError as exc:
        raise RuntimeError(
            "GPT-OSS MoE LoRA requires a Transformers release that provides "
            "Mxfp4Config(dequantize=True). Upgrade Transformers before training."
        ) from exc
    return Mxfp4Config(dequantize=True)


def config_uses_mxfp4_storage(model_config) -> bool:
    """Return whether checkpoint weights are declared as native MXFP4 storage."""
    quantization_config = getattr(model_config, "quantization_config", None)
    if quantization_config is None:
        return False

    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method")
    else:
        quant_method = getattr(quantization_config, "quant_method", None)

    quant_method_value = getattr(quant_method, "value", quant_method)
    return str(quant_method_value).lower() == "mxfp4"


def configure_fsdp_auto_wrap_metadata(model) -> None:
    """Keep only model-present classes in the metadata used by FSDP auto-wrap."""
    declared_class_names = list(dict.fromkeys(getattr(model, "_no_split_modules", None) or []))
    if not declared_class_names:
        raise ValueError(
            "FSDP transformer auto-wrap could not infer a layer class because this model does not define "
            "_no_split_modules. Use a standard Transformers model or provide "
            "transformer_layer_cls_to_wrap explicitly in the FSDP config."
        )

    model_class_names = {module.__class__.__name__ for module in model.modules()}
    inferred_class_names = [name for name in declared_class_names if name in model_class_names]
    ignored_class_names = [name for name in declared_class_names if name not in model_class_names]

    if not inferred_class_names:
        raise ValueError(
            "FSDP transformer auto-wrap could not find any of the model's declared _no_split_modules "
            f"in the loaded module tree. Declared classes: {declared_class_names}."
        )

    # Some model families share metadata between text-only and multimodal model
    # variants. For example, Gemma 3 causal LMs can advertise optional SigLIP
    # classes that are not instantiated. Accelerate treats every advertised name
    # as mandatory, so sanitize the instance metadata before PEFT/FSDP wrapping.
    model._no_split_modules = inferred_class_names

    print(f"FSDP auto-wrap layer classes inferred from the loaded model: {inferred_class_names}")
    if ignored_class_names:
        print(f"FSDP auto-wrap ignored optional classes absent from the loaded model: {ignored_class_names}")


def promote_latest_adapter_checkpoint(output_dir: Path) -> Path:
    """Copy the final checkpoint's HF-compatible LoRA adapter into output_dir."""
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is None:
        raise RuntimeError(
            "FSDP+LoRA training finished without a checkpoint to promote. "
            "Ensure checkpoint saving is enabled during training."
        )

    checkpoint_dir = Path(checkpoint)
    adapter_weight_names = ("adapter_model.safetensors", "adapter_model.bin")
    adapter_weight_sources = [checkpoint_dir / name for name in adapter_weight_names if (checkpoint_dir / name).is_file()]
    adapter_config_source = checkpoint_dir / "adapter_config.json"

    if not adapter_weight_sources or not adapter_config_source.is_file():
        raise RuntimeError(
            f"Final checkpoint {checkpoint_dir} does not contain an HF-compatible LoRA adapter."
        )

    files_to_promote = adapter_weight_sources + [adapter_config_source]
    for optional_name in ("README.md", "training_args.bin"):
        optional_source = checkpoint_dir / optional_name
        if optional_source.is_file():
            files_to_promote.append(optional_source)

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in files_to_promote:
        destination = output_dir / source.name
        temporary_destination = output_dir / f".{source.name}.tmp"
        shutil.copy2(source, temporary_destination)
        os.replace(temporary_destination, destination)

    return checkpoint_dir


def normalize_message(message: Any) -> Optional[Dict[str, str]]:
    if not isinstance(message, dict):
        return None

    role = message.get("role")
    content = message.get("content")

    if role == "system":
        return None

    if role not in {"user", "assistant"}:
        return None

    if content is None:
        content = ""

    return {
        "role": str(role),
        "content": str(content),
    }


def convert_to_prompt_completion(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a row with messages into conversational prompt/completion.

    We use the final assistant message as completion.
    Everything before it becomes prompt.
    System messages are always discarded.
    """
    raw_messages = example.get("messages", [])
    if raw_messages is None:
        raw_messages = []

    messages: List[Dict[str, str]] = []
    for msg in raw_messages:
        normalized = normalize_message(msg)
        if normalized is not None:
            messages.append(normalized)

    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

    if not assistant_indices:
        return {
            "prompt": [],
            "completion": [],
            "_keep": False,
        }

    last_assistant_idx = assistant_indices[-1]
    prompt = messages[:last_assistant_idx]
    completion = [messages[last_assistant_idx]]

    has_user = any(msg["role"] == "user" for msg in prompt)
    has_assistant_completion = len(completion) == 1 and completion[0]["role"] == "assistant"

    return {
        "prompt": prompt,
        "completion": completion,
        "_keep": bool(has_user and has_assistant_completion),
    }


def load_sft_dataset(
    parquet_file: Path,
    *,
    max_samples: Optional[int],
    seed: int,
    name: str,
) -> Dataset:
    parquet_file = parquet_file.resolve()
    if not parquet_file.exists():
        raise FileNotFoundError(f"{name} parquet file does not exist: {parquet_file}")

    dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")

    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    original_columns = list(dataset.column_names)

    dataset = dataset.map(
        convert_to_prompt_completion,
        remove_columns=original_columns,
        desc=f"Converting {name} to prompt/completion",
    )

    dataset = dataset.filter(lambda x: x["_keep"], desc=f"Filtering empty {name} rows")
    dataset = dataset.remove_columns(["_keep"])

    if len(dataset) == 0:
        raise ValueError(f"No usable examples found in {parquet_file}")

    return dataset


def maybe_install_simple_chat_template(tokenizer, model) -> None:
    if tokenizer.chat_template:
        return

    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


def is_using_simple_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) == SIMPLE_CHAT_TEMPLATE


def render_simple_chat_messages(
    messages: List[Dict[str, str]],
    *,
    eos_token: str,
    append_assistant_prompt: bool = False,
) -> str:
    parts: List[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            parts.append(f"User: {content}\n\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}{eos_token}\n\n")
    if append_assistant_prompt:
        parts.append("Assistant:\n")
    return "".join(parts)


def tokenize_with_explicit_completion_mask(
    dataset: Dataset,
    *,
    tokenizer,
    max_length: int,
    name: str,
) -> Dataset:
    eos_token = tokenizer.eos_token or ""

    def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["prompt"]
        completion = example["completion"]
        if not completion or completion[-1]["role"] != "assistant":
            raise ValueError("Expected exactly one assistant completion message.")

        prompt_text = render_simple_chat_messages(
            prompt,
            eos_token=eos_token,
            append_assistant_prompt=True,
        )
        completion_text = completion[-1]["content"] + eos_token + "\n\n"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + completion_ids)[:max_length]
        completion_mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[:max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

    remove_columns = list(dataset.column_names)
    return dataset.map(
        tokenize_example,
        remove_columns=remove_columns,
        desc=f"Tokenizing {name} with explicit completion mask",
    )


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    report_to = []
    if args.report_to and args.report_to.lower() != "none":
        report_to = [x.strip() for x in args.report_to.split(",") if x.strip()]

    eval_strategy = "steps" if args.eval_file is not None else "no"

    config_kwargs = {
        "output_dir": str(args.output_dir),
        "max_length": args.max_length,
        "packing": args.packing,
        "completion_only_loss": True,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": eval_strategy,
        "save_total_limit": args.save_total_limit,
        "report_to": report_to,
        "run_name": args.run_name,
        "dataloader_num_workers": args.dataloader_num_workers,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "remove_unused_columns": True,
    }

    if args.lr_scheduler_kwargs is not None:
        config_kwargs["lr_scheduler_kwargs"] = args.lr_scheduler_kwargs

    if args.chat_template_path:
        config_kwargs["chat_template_path"] = args.chat_template_path

    if args.eos_token:
        config_kwargs["eos_token"] = args.eos_token

    if args.fsdp:
        config_kwargs["fsdp"] = True
        if args.fsdp_config is not None:
            fsdp_config = args.fsdp_config.resolve()
            if not fsdp_config.is_file():
                raise FileNotFoundError(f"FSDP config does not exist: {fsdp_config}")
            config_kwargs["fsdp_config"] = str(fsdp_config)

    # Compatibility fallback for older transformers/trl versions that still use
    # evaluation_strategy instead of eval_strategy.
    try:
        return SFTConfig(**config_kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise

        config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
        return SFTConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_qlora:
        args.use_lora = True

    if args.fsdp_config is not None and not args.fsdp:
        raise ValueError("--fsdp_config requires --fsdp.")

    if args.fsdp and args.gradient_checkpointing:
        raise ValueError(
            "Do not combine --gradient_checkpointing with --fsdp. "
            "Set activation_checkpointing=true in the FSDP config instead."
        )

    if not args.trust_remote_code and should_force_trust_remote_code(args.model_name_or_path):
        print(
            "Detected a local Hugging Face checkpoint with custom code; "
            "enabling trust_remote_code=True automatically."
        )
        args.trust_remote_code = True

    if args.attn_implementation is None and should_force_safe_attn_implementation(args.model_name_or_path):
        print(
            "Detected local OpenLM custom model package; "
            "forcing attn_implementation='eager' because this custom model does not advertise "
            "Flash Attention 2 or SDPA support to Transformers."
        )
        args.attn_implementation = "eager"

    # Construct TrainingArguments before from_pretrained. In a torchrun process this
    # initializes the process group and enables rank-0-only checkpoint loading for
    # FSDP when cpu_ram_efficient_loading is configured.
    sft_args = build_sft_config(args)

    torch_dtype = get_torch_dtype(args)
    device_map = resolve_device_map(args)
    tokenizer_source = infer_tokenizer_source(args.model_name_or_path, args.tokenizer_name_or_path)

    train_dataset = load_sft_dataset(
        args.train_file,
        max_samples=args.max_train_samples,
        seed=args.seed,
        name="train",
    )

    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = load_sft_dataset(
            args.eval_file,
            max_samples=args.max_eval_samples,
            seed=args.seed,
            name="eval",
        )

    print(f"Train examples: {len(train_dataset):,}")
    if eval_dataset is not None:
        print(f"Eval examples:  {len(eval_dataset):,}")

    if args.debug_print_examples > 0:
        print("\nConverted examples:")
        for i in range(min(args.debug_print_examples, len(train_dataset))):
            row = train_dataset[i]
            print(json.dumps(row, indent=2, ensure_ascii=False))
            print()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_qlora:
        compute_dtype = torch_dtype or torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model_type = str(getattr(model_config, "model_type", "") or "")
    requested_target_parameters = args.lora_target_parameters.strip().lower()
    if args.use_lora:
        lora_target_parameters = resolve_lora_target_parameters(
            args.lora_target_parameters,
            model_type,
        )
    else:
        if requested_target_parameters not in {"", "auto", "none"}:
            raise ValueError("Explicit --lora_target_parameters requires --use_lora.")
        lora_target_parameters = []
    if args.attn_implementation is not None:
        setattr(model_config, "_attn_implementation", args.attn_implementation)
        setattr(model_config, "_attn_implementation_internal", args.attn_implementation)

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "config": model_config,
    }

    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if model_type == "gpt_oss" and lora_target_parameters:
        if quantization_config is not None:
            raise ValueError(
                "GPT-OSS MoE parameter LoRA cannot be combined with --use_qlora. "
                "The native MXFP4 experts must first be dequantized to BF16."
            )
        if config_uses_mxfp4_storage(model_config):
            model_kwargs["quantization_config"] = mxfp4_dequantization_config()
            print(
                "GPT-OSS MoE LoRA enabled: loading native MXFP4 expert weights "
                "dequantized to BF16. This uses substantially more model memory."
            )
        else:
            print(
                "GPT-OSS MoE LoRA enabled: checkpoint is already dequantized; "
                "loading its BF16 expert parameters without an MXFP4 quantizer."
            )

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.fsdp and not sft_args.fsdp_config.get("transformer_layer_cls_to_wrap"):
        configure_fsdp_auto_wrap_metadata(model)

    if args.use_simple_chat_template_if_missing and not tokenizer.chat_template and not args.chat_template_path:
        maybe_install_simple_chat_template(tokenizer, model)

    if not tokenizer.chat_template and not args.chat_template_path:
        raise ValueError(
            "Tokenizer has no chat_template. Either pass --chat_template_path or use "
            "--use_simple_chat_template_if_missing."
        )

    if args.gradient_checkpointing or args.fsdp:
        model.config.use_cache = False

    if args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    peft_config = None
    if args.use_lora:
        lora_config_kwargs = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": parse_lora_target_modules(args.lora_target_modules),
        }
        if lora_target_parameters:
            if "target_parameters" not in inspect.signature(LoraConfig).parameters:
                raise RuntimeError(
                    "This PEFT version does not support LoraConfig(target_parameters=...). "
                    "Install PEFT >= 0.17.1 to train GPT-OSS fused MoE parameters."
                )
            validate_lora_target_parameters(model, lora_target_parameters)
            lora_config_kwargs["target_parameters"] = lora_target_parameters

        peft_config = LoraConfig(
            **lora_config_kwargs,
        )

    if is_using_simple_chat_template(tokenizer):
        print(
            "Using simple fallback chat template; pretokenizing with an explicit completion mask "
            "to avoid TRL prompt/completion boundary mismatch warnings."
        )
        train_dataset = tokenize_with_explicit_completion_mask(
            train_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            name="train",
        )
        if eval_dataset is not None:
            eval_dataset = tokenize_with_explicit_completion_mask(
                eval_dataset,
                tokenizer=tokenizer,
                max_length=args.max_length,
                name="eval",
            )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.fsdp and args.use_lora:
        # Trainer already writes an HF-compatible adapter in the final checkpoint.
        # Calling Trainer.save_model() again under FSDP2 gathers the entire frozen
        # base model onto rank 0 before PEFT filters it, which is redundant and can
        # cause prolonged CPU-memory thrashing for large models.
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            checkpoint_dir = promote_latest_adapter_checkpoint(args.output_dir)
            tokenizer.save_pretrained(str(args.output_dir))
            print(f"Promoted final LoRA adapter from {checkpoint_dir} to {args.output_dir}")
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(str(args.output_dir))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(args.output_dir))

    if trainer.is_world_process_zero():
        print(f"Saved model/adapters to: {args.output_dir}")


if __name__ == "__main__":
    main()
