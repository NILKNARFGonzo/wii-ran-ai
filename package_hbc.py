#!/usr/bin/env python3
"""Package the built DOL in the Homebrew Channel SD-card folder layout."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
APP_DIR = Path("apps") / "wii-ran-ai"
OUTPUT = ROOT / "wii-ran-ai-hbc.zip"
FILES = {
    ROOT / "storybananamind-wii.dol": APP_DIR / "boot.dol",
    ROOT / "meta.xml": APP_DIR / "meta.xml",
    ROOT / "icon.png": APP_DIR / "icon.png",
    ROOT / "LICENSE": APP_DIR / "LICENSE",
    ROOT / "medium-chat-wii-weights.bin": APP_DIR / "medium-chat-wii-weights.bin",
    ROOT / "medium-chat-wii-tokenizer.bin": APP_DIR / "medium-chat-wii-tokenizer.bin",
    ROOT / "nano-continuation-wii-weights.bin": APP_DIR / "nano-continuation-wii-weights.bin",
    ROOT / "nano-continuation-wii-tokenizer.bin": APP_DIR / "nano-continuation-wii-tokenizer.bin",
}


def main():
    missing = [str(path.name) for path in FILES if not path.is_file()]
    if missing:
        raise SystemExit("Missing required release files: " + ", ".join(missing))

    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for source, target in FILES.items():
            archive.write(source, target.as_posix())
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
