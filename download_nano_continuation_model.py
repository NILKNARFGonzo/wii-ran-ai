#!/usr/bin/env python3
"""Download the pinned BananaMind 2 Nano base checkpoint and tokenizer."""

import urllib.request
from pathlib import Path


MODEL_ID = "BananaMind/BananaMind-2-Nano"
REVISION = "c8564d1bd3f6177221ed7e4f63ae5f281a677a1c"
FILES = ("config.json", "model.safetensors", "tokenizer.json")
ROOT = Path(__file__).resolve().parent


def main():
    for filename in FILES:
        target = ROOT / f"nano-continuation-{filename}"
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
