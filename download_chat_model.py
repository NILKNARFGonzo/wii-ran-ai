#!/usr/bin/env python3
"""Download the pinned BananaMind 2 Nano Chat checkpoint files."""

import urllib.request
from pathlib import Path


MODEL_ID = "BananaMind/BananaMind-2-Nano-Chat"
REVISION = "6def7ab4d1bd8449d9b75b62bfb174f73d3a8ad6"
FILES = ("config.json", "model.safetensors", "tokenizer.json")
ROOT = Path(__file__).resolve().parent


def main():
    for filename in FILES:
        target = ROOT / f"chat-{filename}"
        if target.is_file() and target.stat().st_size:
            print(f"Using existing {target}")
            continue
        url = (
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
            f"{filename}?download=true"
        )
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {filename} from {MODEL_ID}@{REVISION}...")
        request = urllib.request.Request(url, headers={"User-Agent": "wii-ran-ai"})
        try:
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
