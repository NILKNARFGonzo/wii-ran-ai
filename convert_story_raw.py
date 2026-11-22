import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model.safetensors"
OUTPUT = ROOT / "story-wii-raw"
WEIGHTS_FILE = ROOT / "story-wii-weights.bin"
TOKENIZER_FILE = ROOT / "story-wii-tokenizer.bin"


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


def write_model_package(config, tensors):
    with WEIGHTS_FILE.open("wb") as out:
        out.write(b"SWM1")
        out.write(struct.pack(
            ">7I", 1, config["n_layer"], config["vocab_size"],
            config["n_positions"], config["n_embd"], config["n_head"],
            len(tensors),
        ))

        for name, tensor in tensors:
            is_stability_tensor = (
                ".ln_" in name
                or name.endswith("ln_f.weight")
                or name.endswith("ln_f.bias")
            )
            if is_stability_tensor:
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
            out.write(struct.pack(">HBB", len(name_bytes), dtype, len(dims)))
            out.write(name_bytes)
            for dim in dims:
                out.write(struct.pack(">I", dim))
            out.write(struct.pack(">fI", scale, len(raw)))
            out.write(raw)


def write_tokenizer_package(tokenizer_path):
    tokenizer = json.loads(tokenizer_path.read_text())
    model = tokenizer["model"]
    vocab = model["vocab"]
    mapping = bytelevel_map()
    id_to_bytes = [None] * len(vocab)
    for token, token_id in vocab.items():
        id_to_bytes[token_id] = (
            b"" if token == model.get("unk_token") else bytelevel_token(token, mapping)
        )

    token_to_id = {token: token_id for token, token_id in vocab.items()}
    merges = []
    for left, right in model["merges"]:
        result = left + right
        if left not in token_to_id or right not in token_to_id or result not in token_to_id:
            raise ValueError(f"Tokenizer merge is missing a vocabulary token: {left!r} + {right!r}")
        merges.append((token_to_id[left], token_to_id[right], token_to_id[result]))

    with TOKENIZER_FILE.open("wb") as out:
        out.write(b"TKB1")
        out.write(struct.pack(">II", len(id_to_bytes), len(merges)))
        for token_bytes in id_to_bytes:
            if token_bytes is None or len(token_bytes) > 0xFFFF:
                raise ValueError("Invalid tokenizer vocabulary")
            out.write(struct.pack(">H", len(token_bytes)))
            out.write(token_bytes)
        for left, right, result in merges:
            out.write(struct.pack(">HHH", left, right, result))


def quantize_i8(tensor):
    tensor = tensor.detach().cpu().float()
    flat = tensor.flatten()
    scale = flat.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized.numpy(), scale


def safe_filename(name):
    return name.replace("/", "__").replace(".", "_") + ".bin"


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()

    config = json.loads((ROOT / "config.json").read_text())
    manifest = {
        "format": "story-wii-raw-v1",
        "byte_order": "big",
        "config": config,
        "tensors": [],
    }
    original_tensors = []

    with safe_open(SOURCE, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            original_tensors.append((name, tensor))

            is_stability_tensor = (
                ".ln_" in name
                or name.endswith("ln_f.weight")
                or name.endswith("ln_f.bias")
            )

            if is_stability_tensor:
                dtype = "F32"
                scale = 1.0
                values = tensor.detach().cpu().float().numpy().astype(">f4")
            else:
                dtype = "I8"
                values, scale = quantize_i8(tensor)
                values = values.astype(np.int8)

            filename = safe_filename(name)
            (OUTPUT / filename).write_bytes(values.tobytes(order="C"))
            manifest["tensors"].append({
                "name": name,
                "file": filename,
                "dtype": dtype,
                "shape": list(tensor.shape),
                "scale": scale,
                "elements": tensor.numel(),
            })

    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(ROOT / "config.json", OUTPUT / "config.json")
    shutil.copy2(ROOT / "tokenizer.json", OUTPUT / "tokenizer.json")
    write_model_package(config, original_tensors)
    write_tokenizer_package(ROOT / "tokenizer.json")

    total = sum((OUTPUT / item["file"]).stat().st_size for item in manifest["tensors"])
    print(f"Wrote {len(manifest['tensors'])} tensors")
    print(f"Tensor payload size: {total:,} bytes")
    print(f"Wii weights: {WEIGHTS_FILE.stat().st_size:,} bytes")
    print(f"Wii tokenizer: {TOKENIZER_FILE.stat().st_size:,} bytes")
    print(f"Directory: {OUTPUT}")


if __name__ == "__main__":
    main()
