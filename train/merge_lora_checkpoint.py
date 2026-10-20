#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


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


TOKENIZER_ARTIFACTS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def has_local_tokenizer_artifacts(path: Path) -> bool:
    return path.exists() and path.is_dir() and any((path / name).exists() for name in TOKENIZER_ARTIFACTS)


def infer_tokenizer_source(
    model_name_or_path: str,
    adapter_path: Path,
    explicit_tokenizer_name_or_path: Optional[str],
) -> str:
    if explicit_tokenizer_name_or_path:
        return explicit_tokenizer_name_or_path

    path = Path(model_name_or_path)
    if has_local_tokenizer_artifacts(path):
        return model_name_or_path

    if has_local_tokenizer_artifacts(adapter_path):
        print(f"No tokenizer artifacts found in base model directory; using local tokenizer from adapter at {adapter_path}.")
        return str(adapter_path)

    if not path.exists() or not path.is_dir():
        return model_name_or_path

    config_yaml_path = path / "config.yaml"
    if config_yaml_path.exists():
        try:
            config_yaml = yaml.safe_load(config_yaml_path.read_text())
            tokenizer_name_or_path = (config_yaml.get("tokenizer") or {}).get("tokenizer_name_or_path")
            if tokenizer_name_or_path:
                print(
                    "No tokenizer artifacts found in base model directory; "
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


def get_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None or dtype_name == "auto":
        return None
    return getattr(torch, dtype_name)


def import_peft_model(disable_bitsandbytes: bool):
    if not disable_bitsandbytes:
        from peft import PeftModel

        return PeftModel

    real_find_spec = importlib.util.find_spec

    def patched_find_spec(name, package=None):
        if name == "bitsandbytes":
            return None
        return real_find_spec(name, package)

    # PEFT only needs plain LoRA layers for this merge path, so masking a broken
    # optional bitsandbytes install avoids importing GPU quantization backends.
    importlib.util.find_spec = patched_find_spec
    try:
        from peft import PeftModel
    finally:
        importlib.util.find_spec = real_find_spec

    return PeftModel


def resolve_base_model_path(adapter_path: Path, explicit_base_model: Optional[str]) -> str:
    if explicit_base_model:
        return explicit_base_model

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Could not find adapter_config.json under adapter path: {adapter_path}"
        )

    config = json.loads(adapter_config_path.read_text())
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"adapter_config.json at {adapter_config_path} does not contain base_model_name_or_path"
        )
    return str(base_model)


def copy_optional_artifacts(src_dir: Path, dst_dir: Path) -> None:
    for name in [
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ]:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter checkpoint into its base causal LM and save a standalone HF model."
    )
    parser.add_argument(
        "--adapter_model_path",
        type=Path,
        required=True,
        help="Path to the LoRA adapter directory, e.g. outputs/... or outputs/.../checkpoint-180",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the merged standalone model will be written.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Optional explicit base model path override. Defaults to adapter_config.json.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Optional tokenizer override. Defaults to base model tokenizer fallback logic.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Load/merge dtype. Use float32 if you want the safest merge path on CPU.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default=None,
        choices=[None, "float16", "bfloat16", "float32"],
        help="Optional dtype cast before saving the merged model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to place the base model on during merge, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Save in safetensors format.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        dest="safe_serialization",
        action="store_false",
        help="Disable safetensors output.",
    )
    parser.add_argument(
        "--allow_bitsandbytes_import",
        action="store_true",
        help="Allow PEFT to probe bitsandbytes during import. Leave unset for safer CPU merges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = args.adapter_model_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name_or_path = resolve_base_model_path(adapter_path, args.base_model_name_or_path)

    if not args.trust_remote_code and should_force_trust_remote_code(base_model_name_or_path):
        print(
            "Detected a local Hugging Face checkpoint with custom code; "
            "enabling trust_remote_code=True automatically."
        )
        args.trust_remote_code = True

    if args.attn_implementation is None and should_force_safe_attn_implementation(base_model_name_or_path):
        print(
            "Detected local OpenLM custom model package; "
            "forcing attn_implementation='eager' for merge-time loading."
        )
        args.attn_implementation = "eager"

    tokenizer_source = infer_tokenizer_source(
        base_model_name_or_path,
        adapter_path,
        args.tokenizer_name_or_path,
    )
    load_dtype = get_torch_dtype(args.dtype)
    save_dtype = get_torch_dtype(args.save_dtype)
    disable_bitsandbytes = (args.device == "cpu") and not args.allow_bitsandbytes_import
    if disable_bitsandbytes:
        print("Masking optional bitsandbytes import for CPU LoRA merge.")
    PeftModel = import_peft_model(disable_bitsandbytes=disable_bitsandbytes)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    processor = None
    if getattr(model_config, "vision_config", None) is not None:
        print(f"Loading multimodal processor from: {base_model_name_or_path}")
        processor = AutoProcessor.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    if args.attn_implementation is not None:
        setattr(model_config, "_attn_implementation", args.attn_implementation)
        setattr(model_config, "_attn_implementation_internal", args.attn_implementation)

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "config": model_config,
    }
    if load_dtype is not None:
        model_kwargs["dtype"] = load_dtype

    print(f"Loading base model from: {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )
    base_model.to(args.device)
    base_model.eval()

    print(f"Loading adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )

    print("Merging LoRA weights into the base model")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    if save_dtype is not None:
        print(f"Casting merged model to {args.save_dtype} before saving")
        merged_model.to(save_dtype)

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=args.safe_serialization,
        # GPT-OSS is loaded by dequantizing its MXFP4 expert weights. In
        # transformers 5.12, reversing that load-time conversion while saving
        # collapses every layer's expert weights into the literal keys
        # ``down_proj$`` and ``gate_up_proj$``. Save the in-memory HF names so
        # each layer keeps its own expert tensors and vLLM can load the result.
        save_original_format=model_config.model_type != "gpt_oss",
    )
    tokenizer.save_pretrained(str(output_dir))
    if processor is not None:
        processor.save_pretrained(str(output_dir))
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            image_processor.save_pretrained(str(output_dir))

    copy_optional_artifacts(Path(base_model_name_or_path), output_dir)
    copy_optional_artifacts(adapter_path, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
