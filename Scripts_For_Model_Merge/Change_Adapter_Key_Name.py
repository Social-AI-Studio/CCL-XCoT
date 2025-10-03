#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Dict, Any

import torch

try:
    from safetensors.torch import load_file as st_load_file, save_file as st_save_file
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def load_weights(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if not HAS_SAFETENSORS:
            raise RuntimeError("检测到 .safetensors 但未安装 safetensors，请先 `pip install safetensors`。")
        return st_load_file(path, device="cpu")
    # 默认按 PyTorch 二进制权重读取
    return torch.load(path, map_location="cpu")


def save_weights(state: Dict[str, Any], path: str, force: bool = False):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"目标文件已存在：{path}（使用 --force 覆盖）")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if not HAS_SAFETENSORS:
            raise RuntimeError("目标为 .safetensors 但未安装 safetensors，请先 `pip install safetensors`。")
        st_save_file(state, path)
    else:
        torch.save(state, path)


def transform_keys(
    weights: Dict[str, Any],
    start_from: str = "base_model",
    remove_token: str = ".default",
) -> Dict[str, Any]:
    updated = {}
    for k, v in weights.items():
        new_k = k
        if start_from and start_from in new_k:
            idx = new_k.index(start_from)
            new_k = new_k[idx:]
        if remove_token:
            new_k = new_k.replace(remove_token, "")
        updated[new_k] = v
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="修正适配器权重的键名（可指定起始子串与移除子串），支持 .bin / .safetensors。"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入适配器权重文件路径（例如 adapter_model.bin 或 adapter_model.safetensors）"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出文件路径（默认：与输入同目录，文件名加 _fixed 后缀）"
    )
    parser.add_argument(
        "--start-from",
        default="base_model",
        help="仅保留从该子串开始的键名（默认：base_model；设为空字符串可禁用）"
    )
    parser.add_argument(
        "--remove-token",
        default=".default",
        help="从键名里移除的子串（默认：.default；设为空字符串可禁用）"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="预览打印的键名数量（默认：10）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览不保存"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如目标文件已存在则强制覆盖"
    )
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"❌ 找不到输入文件：{in_path}", file=sys.stderr)
        sys.exit(1)

    # 推断默认输出路径
    if args.output is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_fixed{ext or '.bin'}"
    else:
        out_path = args.output

    # 读取
    weights = load_weights(in_path)

    # 打印原始键名（预览）
    print("原始适配器键名（前 {} 个）:".format(args.preview))
    for k in list(weights.keys())[: args.preview]:
        print("  ", k)

    # 变换
    updated = transform_keys(
        weights,
        start_from=args.start_from,
        remove_token=args.remove_token,
    )

    # 打印修改后键名（预览）
    print("\n修改后适配器键名（前 {} 个）:".format(args.preview))
    for k in list(updated.keys())[: args.preview]:
        print("  ", k)

    if args.dry_run:
        print("\n🛈 dry-run 模式：不写出文件。")
        return

    # 保存
    save_weights(updated, out_path, force=args.force)
    print(f"\n✅ 已保存到：{out_path}")


if __name__ == "__main__":
    main()
