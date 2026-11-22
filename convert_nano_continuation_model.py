#!/usr/bin/env python3
"""Quantize and package the pinned BananaMind 2 Nano base checkpoint."""

from pathlib import Path

from convert_chat_model import convert_model


ROOT = Path(__file__).resolve().parent
CONTEXT = 2048


def main():
    convert_model(
        ROOT / "nano-continuation-config.json",
        ROOT / "nano-continuation-model.safetensors",
        ROOT / "nano-continuation-tokenizer.json",
        ROOT / "nano-continuation-wii-weights.bin",
        ROOT / "nano-continuation-wii-tokenizer.bin",
        "bananamind2_nano",
        CONTEXT,
    )


if __name__ == "__main__":
    main()
