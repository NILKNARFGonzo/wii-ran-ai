#!/usr/bin/env python3
"""Quantize and package the pinned BananaMind 2 Nano Chat checkpoint."""

import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "chat-model.safetensors"
CONFIG = ROOT / "chat-config.json"
TOKENIZER = ROOT / "chat-tokenizer.json"
WEIGHTS_FILE = ROOT / "chat-wii-weights.bin"
TOKENIZER_FILE = ROOT / "chat-wii-tokenizer.bin"
WII_CONTEXT = 2048


def bytelevel_map():
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAD))
    visible += list(range(0xAE, 0x100))
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(codepoints, visible))


def bytelevel_token(token, mapping):
    return bytes(mapping[ord(char)] for char in token)


def quantize_i8(tensor):
    tensor = tensor.detach().cpu().float()
    scale = tensor.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    values = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return values.numpy(), scale


def is_norm_tensor(name):
    return (
        name.endswith(".ln_1.weight")
        or name.endswith(".ln_2.weight")
        or name.endswith(".ln_f.weight")
        or name.endswith(".q_norm.weight")
        or name.endswith(".k_norm.weight")
    )


def write_model_package(config, tensor_items, output_path, context):
    with output_path.open("wb") as output:
        output.write(b"BNM1")
        output.write(struct.pack(
            ">9I2f3I",
            1,
            config["num_hidden_layers"],
            config["vocab_size"],
            context,
            config["hidden_size"],
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["intermediate_size"],
            config["head_dim"],
            config["rope_theta"],
            config["rms_norm_eps"],
            config["bos_token_id"],
            config["eos_token_id"],
            len(tensor_items),
        ))
        for name, tensor in tensor_items:
            if is_norm_tensor(name):
                dtype = 1
                scale = 1.0
                values = tensor.detach().cpu().float().numpy().astype(">f4")
            else:
                dtype = 2
                values, scale = quantize_i8(tensor)
                values = values.astype(np.int8)

            name_bytes = name.encode("utf-8")
            dims = tuple(tensor.shape)
            raw = values.tobytes(order="C")
            output.write(struct.pack(">HBB", len(name_bytes), dtype, len(dims)))
            output.write(name_bytes)
            for dim in dims:
                output.write(struct.pack(">I", dim))
            output.write(struct.pack(">fI", scale, len(raw)))
            output.write(raw)


def write_tokenizer_package(tokenizer_data, config, output_path):
    model = tokenizer_data["model"]
    vocab = model["vocab"]
    if len(vocab) > 0xFFFF:
        raise ValueError("BananaMind tokenizer vocabulary exceeds 16-bit token IDs")
    mapping = bytelevel_map()
    id_to_text = [None] * len(vocab)
    for token, token_id in vocab.items():
        id_to_text[token_id] = token

    special_ids = {
        item["id"]
        for item in tokenizer_data.get("added_tokens", [])
        if item.get("special")
    }
    special_ids.update(
        token_id for token, token_id in vocab.items()
        if token.startswith("<|") and token.endswith("|>")
    )
    token_to_id = {token: token_id for token, token_id in vocab.items()}
    merges = []
    for left, right in model["merges"]:
        result = left + right
        if left not in token_to_id or right not in token_to_id or result not in token_to_id:
            raise ValueError(f"Missing BPE merge token: {left!r} + {right!r}")
        merges.append((token_to_id[left], token_to_id[right], token_to_id[result]))

    with output_path.open("wb") as output:
        output.write(b"TKB2")
        output.write(struct.pack(
            ">4I", len(id_to_text), len(merges), len(special_ids),
            config["eos_token_id"],
        ))
        for token_id, token in enumerate(id_to_text):
            if token is None:
                raise ValueError(f"Tokenizer is missing vocabulary ID {token_id}")
            is_special = token_id in special_ids
            token_bytes = (
                token.encode("utf-8") if is_special
                else bytelevel_token(token, mapping)
            )
            if len(token_bytes) > 0xFFFF:
                raise ValueError("Tokenizer token exceeds 16-bit byte length")
            output.write(struct.pack(">BH", int(is_special), len(token_bytes)))
            output.write(token_bytes)
        for left, right, result in merges:
            output.write(struct.pack(">HHH", left, right, result))


def convert_model(config_path, source_path, tokenizer_path, weights_path,
                  packed_tokenizer_path, expected_model_type, context):
    config = json.loads(config_path.read_text())
    tokenizer_data = json.loads(tokenizer_path.read_text())
    if (config["model_type"] != expected_model_type or
            config["num_attention_heads"] % config["num_key_value_heads"]):
        raise ValueError(f"Unsupported BananaMind architecture: {config['model_type']}")
    if tokenizer_data["model"]["type"] != "BPE":
        raise ValueError("Expected the pinned BananaMind BPE tokenizer")
    pretokenizers = tokenizer_data["pre_tokenizer"]["pretokenizers"]
    if not any(item.get("pattern", {}).get("Regex") == r"\d"
               for item in pretokenizers if item["type"] == "Split"):
        raise ValueError("Expected the digit-isolating BananaMind pre-tokenizer")

    tensors = []
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if name.endswith(".weight") and name.startswith("transformer.h.") and \
                    any(suffix in name for suffix in (
                        ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
                        ".w_gate.", ".w_up.", ".w_down.",
                    )):
                tensor = tensor.transpose(0, 1).contiguous()
            tensors.append((name, tensor))

    if not any(name == "transformer.wte.weight" for name, _ in tensors):
        raise ValueError("Checkpoint is missing tied token embeddings")
    write_model_package(config, tensors, weights_path, context)
    write_tokenizer_package(tokenizer_data, config, packed_tokenizer_path)
    parameter_count = sum(tensor.numel() for _, tensor in tensors)
    print(f"Packed {len(tensors)} tensors ({parameter_count:,} parameters)")
    print(f"Context capped to {context} tokens for bounded Wii KV-cache memory")
    print(f"Wii weights: {weights_path.stat().st_size:,} bytes")
    print(f"Wii tokenizer: {packed_tokenizer_path.stat().st_size:,} bytes")


def main():
    convert_model(
        CONFIG, SOURCE, TOKENIZER, WEIGHTS_FILE, TOKENIZER_FILE,
        "bananamind2_nano", WII_CONTEXT,
    )


if __name__ == "__main__":
    main()
