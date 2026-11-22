# Wii Ran AI

Wii Ran AI is a small C++ story generator for the 7.5M-parameter
[StoryBananaMind-7.5M](https://huggingface.co/BananaMind/StoryBananaMind-7.5M)
GPT-2 model. It uses a Wii-compatible incremental CPU path: model weights
are INT8, activations and KV cache are float32, and serialized multi-byte
values are big-endian. The Wii `.dol` embeds both weights and tokenizer, so
it needs no SD-card model files.

## Use on Wii

1. Download [`wii-ran-ai-hbc.zip`](https://github.com/NILKNARFGonzo/wii-ran-ai/releases/latest)
   from Releases and extract it to the root of an SD card. It installs as
   `apps/wii-ran-ai/boot.dol`.
2. Launch **Wii Ran AI** from the Homebrew Channel.
3. Type with a USB keyboard, or use the Wii Remote D-pad to select on-screen
   letters, numbers, and punctuation. Press **A** to type, **B** to delete,
   **Minus** for a space, and **Plus** to generate.
4. Press **A** or **Plus** after a story to enter another prompt. Press **Home**
   to quit.

Prompts are limited to 256 characters and generations to 80 tokens or the
model's 256-token context, whichever comes first. The model is trained for
short English stories, not general question answering.

## Build from source

Install Python 3, PyTorch, NumPy, `safetensors`, devkitPro with the Wii/
devkitPPC and libogc components, then run:

```sh
python3 -m pip install torch numpy safetensors
python3 download_model.py
make
```

`make` converts the pinned upstream checkpoint, builds the Wii DOL, and
creates `wii-ran-ai-hbc.zip`. Extract that ZIP to the root of an SD card.
The embedded model weights and tokenizer are Apache-2.0 from BananaMind.

## Test on a computer

```sh
g++ -O3 -std=c++11 wii_story.cpp -o wii_story_host
./wii_story_host story-wii-weights.bin story-wii-tokenizer.bin prompt.txt 32
```

Arguments are weights path, tokenizer path, prompt text-file path, and the
maximum number of new tokens. The default prompt is used if the prompt file
does not exist.

## Build for Wii

Generation uses top-k sampling with a repetition penalty and repeated-trigram
blocking to reduce the looping that greedy decoding caused. A small built-in
approximation of the tokenizer's GPT-2 byte-level pre-tokenization rules is
used; this is a hardware demonstration rather than a production-quality
Transformers port. Generation is CPU-only and may be slow on Wii hardware.
