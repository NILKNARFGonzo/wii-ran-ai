#!/usr/bin/env python3
"""Download the pinned StoryBananaMind checkpoint files for conversion."""

import urllib.request
from pathlib import Path


MODEL_ID = "BananaMind/StoryBananaMind-7.5M"
REVISION = "b25ca49faaf1abd93915dbc9185c31b978fbf25b"
FILES = ("config.json", "model.safetensors", "tokenizer.json")
ROOT = Path(__file__).resolve().parent


def main():
    for filename in FILES:
        target = ROOT / filename
        url = (
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
            f"{filename}?download=true"
        )
        print(f"Downloading {filename}...")
        request = urllib.request.Request(url, headers={"User-Agent": "wii-ran-ai"})
        with urllib.request.urlopen(request) as response, target.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
