#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DTYPE_MAP = {
    "auto": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into a base CausalLM model and save the merged model."
    )
    p.add_argument("--base_model_path", required=True, help="Base model path/name (e.g., google/gemma-7b).")
    p.add_argument("--adapter_checkpoint_path", required=True, help="Directory of the PEFT/LoRA adapter.")
    p.add_argument("--save_path", required=True, help="Output directory to save the merged model.")
    p.add_argument(
        "--dtype", choices=list(DTYPE_MAP.keys()), default="fp16",
        help="Model dtype: auto | fp16 | bf16 | fp32 (default: fp16)."
    )
    p.add_argument(
        "--device_map", default="auto",
        help='transformers device_map (default "auto"; e.g., "cpu", "cuda:0").'
    )
    p.add_argument(
        "--trust_remote_code", action="store_true",
        help="Enable when loading models that require executing custom code."
    )
    p.add_argument(
        "--tokenizer_fast", action="store_true",
        help="Prefer the fast tokenizer if available."
    )
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.adapter_checkpoint_path):
        raise FileNotFoundError(f"Adapter path does not exist: {args.adapter_checkpoint_path}")
    else:
        print(f"✅ Adapter path found: {args.adapter_checkpoint_path}")

    torch_dtype = DTYPE_MAP[args.dtype]
    dtype_kw = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

    print(f"➡️  Loading base model: {args.base_model_path}")
    print(f"   - dtype: {args.dtype}")
    print(f"   - device_map: {args.device_map}")

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        use_fast=args.tokenizer_fast,
        trust_remote_code=args.trust_remote_code,
    )

    # 2) Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        **dtype_kw,
    )

    # 3) Load LoRA adapter
    print(f"➡️  Loading LoRA adapter: {args.adapter_checkpoint_path}")
    model = PeftModel.from_pretrained(model, args.adapter_checkpoint_path)

    # 4) Merge and unload adapter
    print("➡️  Merging LoRA adapter into the base model (merge_and_unload)...")
    model = model.merge_and_unload()
    print("✅ LoRA adapter merged into the base model.")

    # 5) Save
    os.makedirs(args.save_path, exist_ok=True)
    print(f"➡️  Saving to: {args.save_path}")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"✅ Merged model saved at: {args.save_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
