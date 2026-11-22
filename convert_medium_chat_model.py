#!/usr/bin/env python3
"""Quantize and package the pinned BananaMind 2 Medium Chat checkpoint."""

from pathlib import Path

from convert_chat_model import convert_model


ROOT = Path(__file__).resolve().parent
CONTEXT = 512


def main():
    convert_model(
        ROOT / "medium-chat-config.json",
        ROOT / "medium-chat-model.safetensors",
        ROOT / "medium-chat-tokenizer.json",
        ROOT / "medium-chat-wii-weights.bin",
        ROOT / "medium-chat-wii-tokenizer.bin",
        "bananamind2_medium",
        CONTEXT,
    )


if __name__ == "__main__":
    main()
